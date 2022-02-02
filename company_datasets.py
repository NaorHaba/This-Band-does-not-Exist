import glob
import shutil
import os
import logging
import pickle

import pandas as pd

import re
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List

from utils import SpecialTokens, clean_text

logger = logging.getLogger(__name__)


def _read_in_chunks(stream, chunk_size=1 * 1024 * 1024):
    while True:
        data = stream.read(chunk_size)
        if not data:
            break
        yield data


class Blacklist:
    def __init__(self, blacklist_set):
        self.blacklist_set = blacklist_set

    def merge(self, other):
        self.blacklist_set |= other.blacklist_set
        return self

    def contains(self, word, recursive=True):
        word = word.strip().lower()
        return (
            word in self.blacklist_set
            or re.sub(r"('inc|ltd|llc|llp|')$", "", word) in self.blacklist_set
            or (recursive and all(self.contains(e, recursive=False) for e in word.split()))
            or (recursive and all(self.contains(e, recursive=False) for e in word.split("-")))
        )

    def collapse_hyphens(self):
        self.blacklist_set |= {"".join(e.split()) for e in self.blacklist_set}
        self.blacklist_set |= {"".join(e.split("-")) for e in self.blacklist_set}

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    @classmethod
    def from_text_lines(cls, stream):
        return cls(set(e.strip().lower() for e in stream))

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.blacklist_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.blacklist_set)


def _len_range_overlap(x, y):
    start = max(x[0], y[0])
    end = min(x[-1], y[-1]) + 1
    return max(0, end - start)


def _split_range(splits, split_idx):
    splits_tensor = torch.tensor(splits)
    sum_splits = torch.cumsum(splits_tensor, 0)

    if sum_splits[-1] != 1.0:
        raise RuntimeError(f"Splits must sum to 1 (actual: {sum_splits[-1]})")
    elif split_idx >= len(sum_splits):
        raise RuntimeError(f"Invalid split index {split_idx} (must be less than {len(sum_splits)})")

    if split_idx == 0:
        start_range = 0.0
    else:
        start_range = sum_splits[split_idx - 1]

    end_range = sum_splits[split_idx]

    return start_range, end_range


def _in_split_range(split_range, starting_position):
    start_range, end_range = split_range
    return start_range <= starting_position < end_range


def row_count(file):
    df = pd.read_csv(file, index_col=0, usecols=['industry'], lineterminator='\n')
    return df.shape[0]


def _cache_path(class_name, base_directory, filename, **keys):
    path = [class_name]
    for k, v in keys.items():
        if isinstance(v, str):
            path.append(f"{k}-{v}")
            continue

        try:
            path.append(f"{k}-{'-'.join(str(e) for e in iter(v))}")
            continue
        except TypeError:
            pass

        path.append(f"{k}-{str(v)}")

    path.append(filename)
    return os.path.join(base_directory, "__".join(path))


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        logger.warning("LineByLineTextDataset currently doesn't support 'splits' logic, processing entire data")

        assert os.path.isfile(file_path)
        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class CSVDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0, ), split_idx=0, text_max_length=1000
    ):
        """Reads the data from a csv file using pandas module. requires csv ordered by index starting from 0"""
        self.block_size = args.block_size
        self.text_max_length = text_max_length

        assert os.path.isfile(file_path) or os.path.islink(file_path)
        directory, filename = os.path.split(file_path)

        cached_features_file = _cache_path(
            self.__class__.__name__,
            directory,
            filename,
            model_type=args.model_type,
            splits=splits,
            split_idx=split_idx,
            block_size=self.block_size,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("Loaded {len(self.examples)} features")
        else:
            logger.info(
                f"Cache at {cached_features_file} not found... creating features from dataset file at %s", directory,
            )

            self.examples = []
            split_range = _split_range(splits, split_idx)

            total_rows = row_count(file_path)
            logger.info(
                f"loading csv file with %d number of rows", total_rows
            )

            with pd.read_csv(file_path, chunksize=50000, index_col=0, lineterminator='\n') as reader:
                reached_first = False
                for i, chunk in enumerate(reader):
                    print(f"Reading chunk {i}...")
                    starting_index = chunk.index[0]
                    if i == 0:
                        assert (starting_index == 0), "csv file should start from index 0"
                    if _in_split_range(split_range, starting_index / total_rows):
                        reached_first = True
                        self.examples.extend(self._make_examples(tokenizer, chunk))
                    elif reached_first:
                        break

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _make_examples(self, tokenizer, data: pd.DataFrame):

        def clean_decode_text(row):
            txt = row['text']
            txt = clean_text(txt, self.text_max_length)

            company_name = row['company_name'].replace('-', ' ')  # clean company name seperator '-'

            return ''.join([SpecialTokens.BOS_TOKEN, company_name, SpecialTokens.IND_SEP, row['industry'],
                            SpecialTokens.TEXT_SEP, txt, SpecialTokens.EOS_TOKEN])

        text_batch = data.apply(clean_decode_text, axis=1).tolist()
        return tokenizer(text_batch, add_special_tokens=True, max_length=self.block_size, truncation=True)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    assert (args.line_by_line != args.csv), "received multiple data types, please specify the exact format"
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    elif args.csv:
        return CSVDataset(tokenizer, args, file_path=file_path, splits=args.splits,
                          split_idx=int(args.eval_split_idx if evaluate else args.train_split_idx))
    else:
        raise NotImplementedError("Current implemented Dataset is only LineByLineTextDataset and CSV")


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
