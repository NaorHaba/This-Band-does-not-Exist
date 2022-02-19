import logging
import os
import time
from functools import partial

import requests
from flask import Flask, request, Response, jsonify
from flask_cors import cross_origin
import json

from band_maker.band_generator import GenerationInput, BandGenerator
from band_maker.custom_generate import decrease_temperature_gradually

logger = logging.getLogger(__name__)


def check():
    gen_input = GenerationInput(**json.loads(b'{"band_name":"Shilshul","genre":"Pop","song_name":"Diarrhea"}'))
    logger.info(f"Received input {gen_input}")
    if gen_input.song_name and not gen_input.band_name:
        generator = song_name_forward_generator
    else:
        generator = band_forward_generator
    gen_band = generator.generate_by_input(gen_input,
                                           temperature=2.0,
                                           transform_logits_warper=partial(decrease_temperature_gradually,
                                                                           decrease_factor=0.85),
                                           top_k=300,
                                           num_return_sequences=12,
                                           max_length=1024
                                           )
    print(gen_band)
    # res = jsonify(gen_band)
    # return res


if __name__ == '__main__':
    print(os.getcwd())
    band_forward_generator = BandGenerator('../../models/gpt2_forward_model',
                                           blacklist_path='../../data/artists_blacklist.pickle',
                                           genres_path='../../data/genres.pickle'
                                           )
    song_name_forward_generator = ...
    print('Starting server...')
    check()
