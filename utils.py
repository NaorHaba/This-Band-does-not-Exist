import re
from typing import NamedTuple, List

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, OpenAIGPTConfig, OpenAIGPTLMHeadModel, \
    OpenAIGPTTokenizer, BertConfig, BertForMaskedLM, BertTokenizer, RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, \
    DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer, CamembertConfig, CamembertForMaskedLM, \
    CamembertTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer

# from archive.modeling import GPT2LMHeadWithWeightedLossModel

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    "T5": (T5Config, T5ForConditionalGeneration, T5Tokenizer)
}


class SpecialTokens:
    BOS_TOKEN = "<|bog|>"
    EOS_TOKEN = "<|eog|>"
    PAD = "<|pad|>"

    GNR_SEP = "<|gnr|>"  # genre
    SNG_SEP = "<|sng|>"  # song name
    LRC_SEP = "<|lrc|>"  # lyrics

    @classmethod
    def special_tokens_dict(cls):
        return {
            "bos_token": cls.BOS_TOKEN,
            "eos_token": cls.EOS_TOKEN,
            "pad_token": cls.PAD,
            "additional_special_tokens": [cls.GNR_SEP, cls.LRC_SEP],
        }


class TokenGroup(NamedTuple):
    separator: List[int] = []
    payload: List[int] = []
    remove_if_truncated: bool = False


def _access_zero_assert(item):
    if len(item) != 1:
        raise RuntimeError("Expected length 1 in item")

    return item[0]


def clean_text(text, truncate=1000):
    txt = text
    txt = re.sub(r'[^\x00-\x7F]+', ' ', txt)  # leave only ascii characters
    txt = txt.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\xa0', ' ')  # remove special text operators
    txt = re.sub(' +', ' ', txt)  # remove multiple spaces
    txt = txt[:truncate]  # truncate according to text max length
    return txt
