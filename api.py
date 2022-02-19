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

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return Response("hello")


@app.route('/bands', methods=['POST'])
@cross_origin()
def generate_band():
    gen_input = GenerationInput(**json.loads(request.data))
    logger.info(f"Received input {gen_input}")
    if gen_input.song_name and not gen_input.band_name:
        generator = song_name_forward_generator
    else:
        generator = band_forward_generator
    try:
        gen_band = generator.generate_by_input(gen_input,
                                               temperature=2.0,
                                               transform_logits_warper=partial(decrease_temperature_gradually,
                                                                               decrease_factor=0.85),
                                               top_k=300,
                                               num_return_sequences=12,
                                               max_length=1024
                                               )
        res = jsonify(gen_band)
        return res
    except Exception as e:  # TODO
        logger.error(e)
        return Response("You are already registered!", status=409)


if __name__ == '__main__':
    band_forward_generator = BandGenerator('models/gpt2_forward_model',
                                           blacklist_path='data/artists_blacklist.pickle',
                                           genres_path='data/genres.pickle'
                                           )
    song_name_forward_generator = ...
    print('Starting server...')
    app.run(debug=True)  # TODO port=config.port
