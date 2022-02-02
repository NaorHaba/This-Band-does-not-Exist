import pickle
import re
import sys
import time
from dataclasses import dataclass

import torch
import logging
from tqdm import tqdm

from company_datasets import Blacklist
from transformers import AutoModelWithLMHead, AutoTokenizer

import os
import csv

from utils import SpecialTokens

logger = logging.getLogger(__name__)


class CompanyGenerator:
    def __init__(self, model, tokenizer=None, blacklist_path=None, industries_path=None, device=None):
        assert tokenizer or isinstance(model, str), "if model is not a model path, tokenizer should be provided"

        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device {self.device}")

        # LOAD BLACKLIST
        if blacklist_path and os.path.isfile(blacklist_path):
            logger.info(f"Loading company blacklist from {blacklist_path}...")
            self.blacklist = Blacklist.load(blacklist_path)
            logger.info(f"Loaded {len(self.blacklist)} companies to blacklist")
        else:
            self.blacklist = None

        if industries_path and os.path.isfile(industries_path):
            logger.info(f"Loading industries from {industries_path}...")
            with open(industries_path, 'rb') as f:
                self.industries = pickle.load(f)
            logger.info(f"Loaded {len(self.industries)} industries")
        else:
            self.industries = None

        if isinstance(model, str):
            # LOAD TOKENIZER
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.tokenizer.add_special_tokens(SpecialTokens.special_tokens_dict())
            logger.info("Loaded tokenizer")

            # LOAD FORWARD MODEL

            logger.info(f"Loading forward model from {model}")
            self.forward_model = AutoModelWithLMHead.from_pretrained(model).to(self.device)
            logger.info("Loaded forward model")
        else:
            self.forward_model = model
            self.tokenizer = tokenizer

    @staticmethod
    def _split_re():
        split_re_pat = (
            f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<company>.+?)"
            f"(?:{re.escape(SpecialTokens.IND_SEP)}(?P<industry>.+?))?"
            f"{re.escape(SpecialTokens.TEXT_SEP)}(?P<text>.+?)"
            f"{re.escape(SpecialTokens.EOS_TOKEN)}*"
        )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)
        return split_re

    def evaluate_creativity(self, num_to_generate, max_iteration, max_length=512):
        _, stats = self.generate_companies(num=num_to_generate,
                                           max_iterations=max_iteration,
                                           generation_args=dict(top_k=300,
                                                                num_return_sequences=48,
                                                                max_length=min(max_length, 512),
                                                                do_sample=True)
                                           )
        # calculate weighted average from generation stats
        score = (stats.num_returned + sum([cand.score for cand in stats.viable_candidates])) / stats.num_items_considered

        return {
            "generation_score": score,
            "success_rate": stats.num_returned / stats.num_items_considered,
        }

    def generate_companies(
            self,
            prefix=SpecialTokens.BOS_TOKEN,
            num=100,
            max_iterations=10,
            generation_args: dict = {},
            filter_generated=True,
            dedupe_titles=True,
            user_filter=None,
            min_text_words=3,
            save_path=None
    ):

        # GENERATE COMPANIES:

        start = time.time()
        ret = []
        num_iteration = 0

        # ENCODE PREFIX

        if isinstance(prefix, str):
            input_ids = self.tokenizer.encode(prefix, return_tensors="pt").long().to(self.forward_model.device)
        else:
            input_ids = torch.tensor([prefix], dtype=torch.long).to(self.forward_model.device)

        split_re = self._split_re()
        seen_companies = set()
        stats = GenerationStats()
        t = tqdm(total=num)

        while len(ret) < num and num_iteration < max_iterations:
            current_ret = []
            num_iteration += 1
            stats.num_iterations += 1

            # GENERATION
            generated = self.forward_model.generate(input_ids,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    bos_token_id=self.tokenizer.bos_token_id,
                                                    eos_token_id=self.tokenizer.eos_token_id,
                                                    **generation_args)
            for i in range(generated.size()[0]):
                if (len(ret) + len(current_ret)) >= num:
                    break
                stats.viable_candidates = stats.viable_candidates[:1000]

                stats.num_items_considered += 1
                sentence_tokens = generated[i, :].tolist()
                decoded = self.tokenizer.decode(sentence_tokens)

                m = split_re.match(decoded)
                if not m:
                    stats.num_failed_match += 1
                    continue

                company = m.group("company")  # company name
                industry = m.group("industry")  # industry
                text = m.group("text")  # text

                generated_company = GeneratedCompany(
                    company=company and company.strip(),
                    industry=industry and industry.strip(),
                    text=text and text.strip()
                )

                if filter_generated:
                    stripped_company = re.sub(r"('inc|ltd|llp|')$", "", company)

                    if self.blacklist and self.blacklist.contains(stripped_company):
                        stats.num_blacklist_filtered += 1
                        continue

                    if dedupe_titles and company.strip().lower() in seen_companies:
                        stats.num_seen_filtered += 1
                        continue

                    if len(text.split()) < min_text_words:
                        stats.num_short_texts += 1
                        stats.viable_candidates.append(GeneratedCompanyCandidate(0.2, generated_company))
                        continue

                    if self.industries and industry not in self.industries:
                        stats.num_industry_filter += 1
                        stats.viable_candidates.append(GeneratedCompanyCandidate(0.5, generated_company))
                        continue

                    if stripped_company.lower() not in text.lower():
                        stats.num_text_missing_company += 1
                        stats.viable_candidates.append(GeneratedCompanyCandidate(0.8, generated_company))
                        continue

                    if user_filter and not user_filter(generated_company):
                        stats.num_user_filtered += 1
                        continue

                    t.update()
                    current_ret.append(generated_company)
                    seen_companies.add(company.strip().lower())
                else:
                    t.update()
                    current_ret.append(generated_company)

            if save_path:
                if not os.path.isfile(save_path):
                    f = open(save_path, 'a')
                    f.close()
                with open(save_path, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    if os.stat(save_path).st_size == 0:
                        columns = ['company_name', 'industry', 'text']
                        writer.writerow(columns)
                    generated_companies = current_ret
                    rows = [[c.company, c.industry, c.text] for c in generated_companies]
                    writer.writerows(rows)

            ret += current_ret

        stats.num_returned = len(ret)
        stats.wall_time = time.time() - start

        return ret[:num], stats


@dataclass
class GeneratedCompany:
    company: str
    industry: str
    text: str

    @classmethod
    def print_companies(cls, companies, f=sys.stdout):
        for company in companies:
            company_str = [company.company, f"/{company.industry}/"]
            print(" ".join(company_str), file=f)
            print(f"\t{company.text}", file=f)
            print("----------------", file=f)


@dataclass
class GeneratedCompanyCandidate:
    score: float
    candidate: GeneratedCompany


@dataclass
class GenerationStats:
    num_iterations: int = 0

    num_items_considered: int = 0
    num_failed_match: int = 0
    num_blacklist_filtered: int = 0
    num_seen_filtered: int = 0
    num_industry_filter: int = 0

    num_short_texts: int = 0
    num_text_missing_company: int = 0

    num_user_filtered: int = 0
    num_returned: int = 0
    wall_time: float = 0.0

    viable_candidates = []

    def __str__(self):
        return (
            f"iterations={self.num_iterations} time={self.wall_time} | "
            + ", ".join(
                f"{k} {v / self.num_items_considered:.2f}@{v}"
                for k, v in (
                    ("items_considered", self.num_items_considered),
                    ("failed_match", self.num_failed_match),
                    ("blacklist_filtered", self.num_blacklist_filtered),
                    ("seen_filtered", self.num_seen_filtered),
                    ("short_definitions", self.num_short_texts),
                    ("text_missing_company", self.num_text_missing_company),
                    ("user_filtered", self.num_user_filtered),
                    ("returned", self.num_returned),
                )
            )
        )
