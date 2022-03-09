import random
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

from band_maker.band_datasets import Blacklist
from transformers import AutoModelWithLMHead, AutoTokenizer

import os
import csv

from band_maker.custom_generate import decrease_temperature_gradually, custom_sample
from band_maker.utils import SpecialTokens

logger = logging.getLogger(__name__)


@dataclass
class GeneratedBand:
    band_name: str
    genre: str
    song_name: str
    lyrics: str

    @classmethod
    def print_bands(cls, bands, f=sys.stdout):
        for band in bands:
            band_str = [band.band_name, f"/{band.genre}/"]
            print(" ".join(band_str), file=f)
            print(f"\t{band.song_name}", file=f)
            print(f"\t{band.lyrics}", file=f)
            print("----------------", file=f)


@dataclass
class GeneratedBandCandidate:
    score: float
    candidate: GeneratedBand


@dataclass
class GenerationInput:
    band_name: str
    genre: str
    song_name: str

    def prepare_input(self):
        reverse = False
        prefix = SpecialTokens.BOS_TOKEN
        if self.band_name:
            prefix += self.band_name
            prefix += SpecialTokens.GNR_SEP
            if self.genre:
                prefix += self.genre
                prefix += SpecialTokens.SNG_SEP
                if self.song_name:
                    prefix += self.song_name
                    prefix += SpecialTokens.LRC_SEP
            elif self.song_name:
                prefix += random.sample(["Country", "Electronic", "Folk", "Hip-Hop", "Indie", "Jazz", "Metal", "Pop",
                                         "R&B", "Rock"], 1)
                prefix += SpecialTokens.SNG_SEP
                prefix += self.song_name
                prefix += SpecialTokens.LRC_SEP
        elif self.song_name:
            reverse = True
            prefix += self.song_name
            prefix += SpecialTokens.GNR_SEP
            if self.genre:
                prefix += self.genre
                prefix += SpecialTokens.ART_SEP
        elif self.genre:
            raise RuntimeError("code shouldn't reach here")

        return prefix, reverse


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


class Generator(ABC):
    def __init__(self, model, tokenizer=None, device=None):
        assert tokenizer or isinstance(model, str), "if model is not a model path, tokenizer should be provided"
        self.forward_model, self.tokenizer = self.load_model(model, tokenizer, device)

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
            input_ids,
            **generation_args
    ):

        return custom_sample(self.forward_model, input_ids,
                             pad_token_id=self.tokenizer.pad_token_id,
                             bos_token_id=self.tokenizer.bos_token_id,
                             eos_token_id=self.tokenizer.eos_token_id,
                             **generation_args)


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

        # LOAD GENRES
        if genres_path and os.path.isfile(genres_path):
            logger.info(f"Loading genres from {genres_path}...")
            with open(genres_path, 'rb') as f:
                self.genres = pickle.load(f)
            logger.info(f"Loaded {len(self.genres)} genres")
        else:
            self.genres = None

    @staticmethod
    def _split_re(reverse=False):
        if reverse:
            split_re_pat = (
                f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<song_name>.+?)"
                f"(?:{re.escape(SpecialTokens.GNR_SEP)}(?P<genre>.+?))"
                f"(?:{re.escape(SpecialTokens.ART_SEP)}(?P<band_name>.+?))"
                f"{re.escape(SpecialTokens.LRC_SEP)}(?P<lyrics>.+?)"
                f"{re.escape(SpecialTokens.EOS_TOKEN)}*"
            )
        else:
            split_re_pat = (
                f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<band_name>.+?)"
                f"(?:{re.escape(SpecialTokens.GNR_SEP)}(?P<genre>.+?))"
                f"(?:{re.escape(SpecialTokens.SNG_SEP)}(?P<song_name>.+?))"
                f"{re.escape(SpecialTokens.LRC_SEP)}(?P<lyrics>.+?)"
                f"{re.escape(SpecialTokens.EOS_TOKEN)}*"
            )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)
        return split_re

    def evaluate_creativity(self, num_to_generate, max_iteration, reverse=False, **generation_args):
        _, stats = self.generate_batch(batch_size=num_to_generate,
                                       max_iterations=max_iteration,
                                       reverse=reverse,
                                       **generation_args)

        # calculate weighted average from generation stats
        score = (stats.num_returned + sum([cand.score for cand in stats.viable_candidates])) / stats.num_items_considered

        return {
            "generation_score": score,
            "success_rate": stats.num_returned / stats.num_items_considered,
        }

    def generate_batch(
            self,
            batch_size=100,
            max_iterations=10,
            existing_bands=True,
            seen_bands=True,
            genres=True,
            save_path=None,
            reverse=False,
            **generation_args
    ):

        # GENERATE BANDS:

        if seen_bands:
            seen_bands = set()
        else:
            seen_bands = None
        stats = GenerationStats()

        start = time.time()
        ret = []
        num_iteration = 0

        split_re = self._split_re(reverse)
        t = tqdm(total=batch_size)
        device = self.forward_model.device
        input_ids = self.tokenizer.encode(SpecialTokens.BOS_TOKEN, return_tensors="pt").long().to(device)

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

                band_name = m.group("band_name")  # band name
                genre = m.group("genre")  # genre
                song_name = m.group("song_name")  # song name
                lyrics = m.group("lyrics")  # lyrics

                generated_band = GeneratedBand(
                    band_name=band_name and band_name.strip(),
                    genre=genre and genre.strip(),
                    song_name=song_name and song_name,
                    lyrics=lyrics and lyrics
                )

                tests_succeed = self.pipeline_generated_tests(generated_band, stats, existing_bands,
                                                              seen_bands, genres)
                if not tests_succeed:
                    continue

                t.update()
                current_ret.append(generated_band)
                seen_bands.add(band_name.strip().lower())

                t.update()
                current_ret.append(generated_band)

            if save_path:
                self.save_generation(save_path, current_ret)

            ret += current_ret

        stats.num_returned = len(ret)
        stats.wall_time = time.time() - start

        return ret[:batch_size], stats

    @staticmethod
    def save_generation(save_path, generated_bands):
        if not os.path.isfile(save_path):
            f = open(save_path, 'a')
            f.close()
        with open(save_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            if os.stat(save_path).st_size == 0:
                columns = ['band_name', 'genre', 'lyrics']
                writer.writerow(columns)
            rows = [[b.band_name, b.genre, b.lyrics] for b in generated_bands]
            writer.writerows(rows)

    def generate_by_input(
            self,
            generation_input: GenerationInput,
            max_iterations=10,
            filter_existing_bands=True,
            filter_genres=True,
            **generation_args
    ):

        if generation_input.band_name and filter_existing_bands and self.band_already_exists(generation_input.band_name):
            filter_existing_bands = False
        # GENERATE BANDS:

        prefix, reverse = generation_input.prepare_input()
        split_re = self._split_re(reverse)
        device = self.forward_model.device
        input_ids = self.tokenizer.encode(prefix, return_tensors="pt").long().to(device)

        for _ in range(max_iterations):

            # GENERATION
            generated = self.generate(input_ids, **generation_args)
            decoded_batch = self.tokenizer.batch_decode(generated)
            for decoded in decoded_batch:

                m = split_re.match(decoded)
                if m:
                    band_name = m.group("band_name")  # band name
                    genre = m.group("genre")  # genre
                    song_name = m.group("song_name")  # song name
                    lyrics = m.group("lyrics")  # lyrics

                    generated_band = GeneratedBand(
                        band_name=band_name and band_name.strip(),
                        genre=genre and genre.strip(),
                        song_name=song_name and song_name,
                        lyrics=lyrics and lyrics
                    )

                    tests_succeed = self.pipeline_generated_tests(generated_band,
                                                                  existing_bands=filter_existing_bands,
                                                                  genres=filter_genres)
                    if not tests_succeed:
                        continue
                    else:
                        return generated_band

        # raise CantCreateBandError

    def pipeline_generated_tests(
            self,
            generated_band,
            stats=GenerationStats(),
            existing_bands=False,
            seen_bands=None,
            genres=False
    ):
        band_name, genre, lyrics = generated_band.band_name, generated_band.genre, generated_band.lyrics

        if existing_bands and self.band_already_exists(band_name):
            stats.num_blacklist_filtered += 1
            return False

        if seen_bands and self.seen_band(band_name, seen_bands):
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

    def band_already_exists(self, band_name):
        if not self.existing_bands:
            raise RuntimeError("existing_bands variable doesn't exist")
        return self.existing_bands.contains(band_name)

    def seen_band(self, band_name, seen_bands):
        return band_name.strip().lower() in seen_bands

    @staticmethod
    def song_is_short(lyrics, min_lyrics_words=30):
        return len(lyrics.replace('\n', ' ').split()) < min_lyrics_words

    def genre_not_exist(self, genre):
        if not self.genres:
            raise RuntimeError("genres variable doesn't exist")
        return genre not in self.genres
