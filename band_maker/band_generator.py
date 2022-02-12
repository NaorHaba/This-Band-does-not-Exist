from abc import ABC
import pickle
import re
import sys
import time
from functools import partial

from dataclasses import dataclass

import torch
import logging
from tqdm import tqdm

from band_datasets import Blacklist
from transformers import AutoModelWithLMHead, AutoTokenizer

import os
import csv

from custom_generate import decrease_temperature_gradually, custom_sample
from utils import SpecialTokens

logger = logging.getLogger(__name__)


class Generator(ABC):
    def __init__(self, model, backward_model, tokenizer=None, backward_tokenizer=None, device=None):
        assert tokenizer or isinstance(model, str), "if model is not a model path, tokenizer should be provided"
        self.forward_model, self.tokenizer = self.load_model(model, tokenizer, device)
        # assert backward_tokenizer or isinstance(backward_model,
        #                                         str), "if model is not a model path, tokenizer should be provided"
        self.backward_model, self.backward_tokenizer = self.load_model(backward_model, backward_tokenizer, device)

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
            input_ids=None,
            **generation_args
    ):
        # ENCODE PREFIX
        if input_ids is None:
            input_ids = torch.tensor([SpecialTokens.BOS_TOKEN], dtype=torch.long).to(self.forward_model.device)

        # if not inverse_mode:
        return custom_sample(self.forward_model, input_ids,
                             pad_token_id=self.tokenizer.pad_token_id,
                             bos_token_id=self.tokenizer.bos_token_id,
                             eos_token_id=self.tokenizer.eos_token_id,
                             **generation_args)


@dataclass
class GenerationInput:
    band_name: str
    genre: str
    song_name: str


class BandGenerator(Generator):
    def __init__(self, model, backward_model, tokenizer=None, backward_tokenizer=None,
                 blacklist_path=None, genres_path=None, device=None):
        super().__init__(model, backward_model, tokenizer, backward_tokenizer, device)

        # LOAD BLACKLIST
        if blacklist_path and os.path.isfile(blacklist_path):
            logger.info(f"Loading blacklist from {blacklist_path}...")
            self.existing_bands = Blacklist.load(blacklist_path)
            logger.info(f"Loaded {len(self.existing_bands)} band names to blacklist")
        else:
            self.existing_bands = None

        # LOAD GENRES
        if genres_path and os.path.isfile(genres_path):
            logger.info(f"Loading genres from {genres_path}...")
            with open(genres_path, 'rb') as f:
                self.genres = pickle.load(f)
            logger.info(f"Loaded {len(self.genres)} genres")
        else:
            self.genres = None

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
            generation_input=None,
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

        inverse_mode = generation_input and generation_input.song_name and not generation_input.band_name
        if inverse_mode:
            generation_input = self.generate_by_input(generation_input, 1, max_iterations,
                                                      filter_generated, existing_bands, dedupe_titles, genres,
                                                      user_filter, **generation_args)
        return self.generate_by_input(generation_input, num, max_iterations,
                                      filter_generated, existing_bands, dedupe_titles, genres, user_filter,
                                      save_path, **generation_args)

    def prepare_prefix(self, generation_input: GenerationInput):
        # song name - inverse model
        self.prefix += generation_input.band_name
        if generation_input.genre:
            if generation_input.band_name == '':
                raise RuntimeError("Can't generate genre without a band-name.")
            self.prefix += SpecialTokens.GNR_SEP
            self.prefix += generation_input.genre
        if generation_input.song_name != '':
            self.prefix += SpecialTokens.SNG_SEP
            self.prefix += generation_input.song_name

    def generate_batch(
            self,
            batch_size=100,
            max_iterations=10,
            filter_generated=True,
            existing_bands=True,
            dedupe_titles=True,
            genres=True,
            save_path=None,
            **generation_args
    ):

        # GENERATE BANDS:

        seen_bands = set()
        stats = GenerationStats()

        start = time.time()
        ret = []
        num_iteration = 0

        split_re = self._split_re()
        t = tqdm(total=batch_size)
        device = self.forward_model.device
        input_ids = self.tokenizer.encode([SpecialTokens.BOS_TOKEN], return_tensors="pt").long().to(device)

        while len(ret) < batch_size and num_iteration < max_iterations:
            current_ret = []
            num_iteration += 1
            stats.num_iterations += 1

            # GENERATION
            generated = self.generate(input_ids, **generation_args)
            decoded_batch = self.tokenizer.batch_decode(generated)
            for i, decoded in enumerate(decoded_batch):
                if (len(ret) + len(current_ret)) >= batch_size:
                    break
                stats.viable_candidates = stats.viable_candidates[:1000]
                stats.num_items_considered += 1

                m = split_re.match(decoded)
                if not m:
                    stats.num_failed_match += 1
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

                tests_succeed = self.pipeline_generated_tests(generated_band, stats, existing_bands,
                                                              dedupe_titles, genres)
                if not tests_succeed:
                    continue

                t.update()
                current_ret.append(generated_band)
                seen_bands.add(band.strip().lower())

                t.update()
                current_ret.append(generated_band)

            if save_path:  # TODO add to func
                if not os.path.isfile(save_path):
                    f = open(save_path, 'a')
                    f.close()
                with open(save_path, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    if os.stat(save_path).st_size == 0:
                        columns = ['band_name', 'genre', 'lyrics']
                        writer.writerow(columns)
                    generated_bands = current_ret
                    rows = [[b.band, b.genre, b.lyrics] for b in generated_bands]
                    writer.writerows(rows)

            ret += current_ret

        stats.num_returned = len(ret)
        stats.wall_time = time.time() - start

        return ret[:batch_size], stats

    def generate_by_input(
            self,
            generation_input: GenerationInput = GenerationInput('', '', ''),
            max_iterations=10,
            filter_existing_bands=True,
            filter_genres=True,
            **generation_args
    ):

        assert not (self.band_already_exists(GenerationInput.band_name) and filter_existing_bands), \
            "Can't filter band name for existing band received"
        # GENERATE BANDS:

        split_re = self._split_re()

        prefix = self.prepare_prefix(generation_input)
        device = self.forward_model.device
        input_ids = self.tokenizer.encode(prefix, return_tensors="pt").long().to(device)

        for _ in range(max_iterations):

            # GENERATION
            generated = self.generate(input_ids, **generation_args)
            decoded_batch = self.tokenizer.batch_decode(generated)
            for decoded in decoded_batch:

                m = split_re.match(decoded)
                if m:
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

                    tests_succeed = self.pipeline_generated_tests(generated_band,
                                                                  existing_bands=filter_existing_bands,
                                                                  genres=filter_genres)
                    if not tests_succeed:
                        continue
                    else:
                        return generated_band

        raise CantCreateBandError

    # def generate_inverse(
    #         self,
    #         generation_input=None,
    #         num=1,
    #         max_iterations=10,
    #         filter_generated=True,
    #         existing_bands=True,
    #         dedupe_titles=True,
    #         genres=True,
    #         user_filter=None,
    #         **generation_args
    # ):
    #
    #     # GENERATE BANDS:
    #
    #     num_iteration = 0
    #
    #     split_re = self._split_re(model='inverse')
    #     t = tqdm(total=num)
    #     if not generation_input:
    #         generation_input = GenerationInput('', '', '')
    #     self.prepare_prefix(generation_input)
    #     input_ids = self.tokenizer.encode(self.prefix, return_tensors="pt").long().to(self.forward_model.device)
    #
    #     while num_iteration < max_iterations:
    #         current_ret = []
    #         num_iteration += 1
    #
    #         # GENERATION
    #         generated = self.generate('inverse', input_ids, **generation_args)
    #
    #         for i in range(generated.size()[0]):
    #             if len(current_ret) >= num:
    #                 break
    #
    #             sentence_tokens = generated[i, :].tolist()
    #             decoded = self.tokenizer.decode(sentence_tokens)
    #
    #             m = split_re.match(decoded)
    #             if not m:
    #                 self.stats.num_failed_match += 1
    #                 continue
    #
    #             band = m.group("band")  # band name
    #             genre = m.group("genre")  # genre
    #             song = m.group("song")  # song name
    #
    #             generation_input = GenerationInput(band, genre, song)
    #
    #             generated_band = GeneratedBand(
    #                 band=band and band.strip(),
    #                 genre=genre and genre.strip(),
    #                 song=song and song,
    #                 lyrics=''
    #             )
    #
    #             if filter_generated:
    #                 tests_succeed = self.pipeline_generated_tests(
    #                     generated_band, existing_bands, dedupe_titles, genres, user_filter, generation_input,
    #                     model='inverse')
    #                 if not tests_succeed:
    #                     continue
    #
    #                 t.update()
    #                 current_ret.append(generated_band)
    #                 self.seen_bands.add(band.strip().lower())
    #             else:
    #                 t.update()
    #                 current_ret.append(generated_band)
    #
    #     return generation_input

    def pipeline_generated_tests(
            self,
            generated_band,
            stats=GenerationStats(),
            existing_bands=False,
            dedupe_titles=False,
            genres=False
    ):
        band, genre, lyrics = generated_band.band, generated_band.genre, generated_band.lyrics

        if existing_bands and self.band_already_exists(band):
            stats.num_blacklist_filtered += 1
            return False

        if dedupe_titles and self.seen_band(band):
            stats.num_seen_filtered += 1
            return False

        if self.song_is_short(lyrics):
            stats.num_short_texts += 1
            stats.viable_candidates.append(GeneratedBandCandidate(0.2, generated_band))
            return False

        if genres and self.genre_not_exist(genre):
            stats.num_genre_filter += 1
            stats.viable_candidates.append(GeneratedBandCandidate(0.5, generated_band))
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
        return len(lyrics.replace('\n', ' ').split()) < min_lyrics_words

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
                            ("returned", self.num_returned),
                        )
                    )
                )