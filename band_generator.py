import abc
import pickle
import re
import sys
import time
from typing import Optional, List

from dataclasses import dataclass

import torch
import logging
from tqdm import tqdm

from band_datasets import Blacklist
from transformers import AutoModelWithLMHead, AutoTokenizer

import os
import csv

from utils import SpecialTokens

from abc import ABC

logger = logging.getLogger(__name__)


class Generator(ABC):
    def __init__(self, model, backward_model, tokenizer=None, backward_tokenizer=None, device=None):
        assert tokenizer or isinstance(model, str), "if model is not a model path, tokenizer should be provided"
        self.forward_model, self.tokenizer = self.load_model(model, tokenizer, device)
        # assert backward_tokenizer or isinstance(backward_model,
        #                                         str), "if model is not a model path, tokenizer should be provided"
        # #TODO: is this how we load this model too?
        # self.backward_model, self.backward_tokenizer = self.load_model(backward_model, backward_tokenizer, device)

    @staticmethod
    def load_model(forward_model, tokenizer, device):
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        logger.info(f"Using device {device}")

        if isinstance(forward_model, str):
            # LOAD TOKENIZER
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(forward_model)
            tokenizer.add_special_tokens(SpecialTokens.special_tokens_dict())
            logger.info("Loaded tokenizer")

            # LOAD FORWARD MODEL
            logger.info(f"Loading forward model from {forward_model}")
            forward_model = AutoModelWithLMHead.from_pretrained(forward_model).to(device)
            logger.info("Loaded forward model")

        return forward_model, tokenizer

    def generate(
            self,
            direction='forward',
            input_ids=None,
            **generation_args
    ):
        # ENCODE PREFIX
        # TODO: maybe in backward model this 'if' should be different
        if input_ids is None:
            input_ids = torch.tensor([SpecialTokens.BOS_TOKEN], dtype=torch.long).to(self.forward_model.device)

        if direction == 'forward':
            return self.forward_model.generate(input_ids,
                                               pad_token_id=self.tokenizer.pad_token_id,
                                               bos_token_id=self.tokenizer.bos_token_id,
                                               eos_token_id=self.tokenizer.eos_token_id,
                                               **generation_args)
        # else:
        #     return self.backward_model.generate(input_ids,
        #                                         pad_token_id=self.backward_tokenizer.pad_token_id,
        #                                         bos_token_id=self.backward_tokenizer.bos_token_id,
        #                                         eos_token_id=self.backward_tokenizer.eos_token_id,
        #                                         **generation_args)


@dataclass
class GenerationInput:
    band_name: str
    genre: str
    song_name: str


class BandGenerator(Generator):
    def __init__(self, model, tokenizer=None, blacklist_path=None, genres_path=None, device=None):
        super().__init__(model, tokenizer, device)

        # LOAD BLACKLIST
        if blacklist_path and os.path.isfile(blacklist_path):
            logger.info(f"Loading blacklist from {blacklist_path}...")
            self.existing_bands = Blacklist.load(blacklist_path)
            logger.info(f"Loaded {len(self.existing_bands)} band names to blacklist")
        else:
            self.existing_bands = None

        if genres_path and os.path.isfile(genres_path):
            logger.info(f"Loading genres from {genres_path}...")
            with open(genres_path, 'rb') as f:
                self.genres = pickle.load(f)
            logger.info(f"Loaded {len(self.genres)} genres")
        else:
            self.genres = None

        self.seen_bands = set()
        self.stats = GenerationStats()
        self.prefix = SpecialTokens.BOS_TOKEN

    @staticmethod
    def _split_re():
        split_re_pat = (
            f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<band>.+?)"
            f"(?:{re.escape(SpecialTokens.GNR_SEP)}(?P<genre>.+?))"
            f"(?:{re.escape(SpecialTokens.SNG_SEP)}(?P<song>.+?))"
            f"{re.escape(SpecialTokens.LRC_SEP)}(?P<lyrics>.+?)"
            f"{re.escape(SpecialTokens.EOS_TOKEN)}*"
        )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)
        return split_re

    # TODO add "generation args" and pass with **
    def evaluate_creativity(self, num_to_generate, max_iteration, max_length=1024):
        gen = self.generate_by_input(num=num_to_generate,
                                     max_iterations=max_iteration,
                                     generation_args=dict(top_k=300,
                                                          num_return_sequences=12,
                                                          max_length=min(max_length, self.tokenizer.model_max_length),
                                                          do_sample=True)
                                     )

        # calculate weighted average from generation stats
        score = (self.stats.num_returned + sum([cand.score for cand in self.stats.viable_candidates])) \
                / self.stats.num_items_considered

        return {
            "generation_score": score,
            "success_rate": self.stats.num_returned / self.stats.num_items_considered,
        }

    def generate_band(
            self,
            generation_input: GenerationInput,
            num=100,
            max_iterations=10,
            generation_args: dict = {},
            filter_generated=True,
            existing_bands=True,
            dedupe_titles=True,
            genres=True,
            user_filter=None,
            save_path=None
    ):
        prefix = SpecialTokens.BOS_TOKEN
        return self.generate_by_input()

    def prepare_prefix(self, input: GenerationInput):
        # TODO: How to add genre without band_name? - error
        # song name - inverse model
        self.prefix += input.band_name
        if input.genre != '':
            if input.band_name == '':
                raise RuntimeError("Can't generate genre without a band-name.")
            self.prefix += SpecialTokens.GNR_SEP
            self.prefix += input.genre
        if input.song_name != '':
            self.prefix += SpecialTokens.SNG_SEP
            self.prefix += input.song_name

    def generate_by_input(
            self,
            input=None,
            num=100,
            max_iterations=10,
            filter_generated=True,
            existing_bands=True,
            dedupe_titles=True,
            genres=True,
            user_filter=None,
            save_path=None,
            **generation_args
    ):

        # GENERATE BANDS:

        start = time.time()
        ret = []
        num_iteration = 0

        split_re = self._split_re()
        t = tqdm(total=num)
        if not input:
            input = GenerationInput('', '', '')
        self.prepare_prefix(input)
        input_ids = self.tokenizer.encode(self.prefix, return_tensors="pt").long().to(self.forward_model.device)

        while len(ret) < num and num_iteration < max_iterations:
            current_ret = []
            num_iteration += 1
            self.stats.num_iterations += 1

            # GENERATION
            generated = self.generate('forward', input_ids, **generation_args)

            for i in range(generated.size()[0]):
                if (len(ret) + len(current_ret)) >= num:
                    break
                self.stats.viable_candidates = self.stats.viable_candidates[:1000]

                self.stats.num_items_considered += 1
                sentence_tokens = generated[i, :].tolist()
                decoded = self.tokenizer.decode(sentence_tokens)

                m = split_re.match(decoded)
                if not m:
                    self.stats.num_failed_match += 1
                    continue

                band = m.group("band")  # band name
                genre = m.group("genre")  # genre
                song = m.group("song")  # song name
                lyrics = m.group("lyrics")  # lyrics

                generated_band = GeneratedBand(
                    band=band and band.strip(),
                    genre=genre and genre.strip(),
                    song=song and song,
                    lyrics=lyrics and lyrics
                )

                if filter_generated:
                    tests_succeed = self.pipeline_generated_tests(
                        generated_band, existing_bands, dedupe_titles, genres, user_filter, input)
                    if not tests_succeed:
                        continue

                    t.update()
                    current_ret.append(generated_band)
                    self.seen_bands.add(band.strip().lower())
                else:
                    t.update()
                    current_ret.append(generated_band)

            if save_path:
                if not os.path.isfile(save_path):
                    f = open(save_path, 'a')
                    f.close()
                with open(save_path, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    if os.stat(save_path).st_size == 0:
                        columns = ['band_name', 'genre', 'lyrics']
                        writer.writerow(columns)
                    generated_companies = current_ret
                    rows = [[c.band, c.genre, c.lyrics] for c in generated_companies]
                    writer.writerows(rows)

            ret += current_ret

        self.stats.num_returned = len(ret)
        self.stats.wall_time = time.time() - start

        return ret[:num]

    def pipeline_generated_tests(self, generated_band, existing_bands, dedupe_titles, genres, user_filter, input):
        band, genre, lyrics = generated_band.band, generated_band.genre, generated_band.lyrics

        if existing_bands and self.band_already_exists(band) and input.band_name == '':
            self.stats.num_blacklist_filtered += 1
            return False

        if dedupe_titles and self.seen_band(band):
            self.stats.num_seen_filtered += 1
            return False

        if self.song_is_short(lyrics):
            self.stats.num_short_texts += 1
            self.stats.viable_candidates.append(GeneratedBandCandidate(0.2, generated_band))
            return False

        if genres and self.genre_not_exist(genre):
            self.stats.num_genre_filter += 1
            self.stats.viable_candidates.append(GeneratedBandCandidate(0.5, generated_band))
            return False

        if user_filter and not user_filter(generated_band):
            self.stats.num_user_filtered += 1
            return False

        return True

    def band_already_exists(self, band):
        if not self.existing_bands:
            raise RuntimeError("existing_bands variable doesn't exist")
        return self.existing_bands.contains(band)

    def seen_band(self, band):
        return band.strip().lower() in self.seen_bands

    @staticmethod
    def song_is_short(lyrics, min_lyrics_words=30):
        return len(lyrics.split()) < min_lyrics_words

    def genre_not_exist(self, genre):
        if not self.genres:
            raise RuntimeError("genres variable doesn't exist")
        return genre not in self.genres


@dataclass
class GeneratedBand:
    band: str
    genre: str
    song: str
    lyrics: str

    @classmethod
    def print_bands(cls, bands, f=sys.stdout):
        for band in bands:
            band_str = [band.band, f"/{band.genre}/"]
            print(" ".join(band_str), file=f)
            print(f"\t{band.song}", file=f)
            print(f"\t{band.lyrics}", file=f)
            print("----------------", file=f)


@dataclass
class GeneratedBandCandidate:
    score: float
    candidate: GeneratedBand


@dataclass
class GenerationStats:
    num_iterations: int = 0

    num_items_considered: int = 0
    num_failed_match: int = 0
    num_blacklist_filtered: int = 0
    num_seen_filtered: int = 0
    num_genre_filter: int = 0

    num_short_texts: int = 0

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
                ("user_filtered", self.num_user_filtered),
                ("returned", self.num_returned),
            )
        )
        )
