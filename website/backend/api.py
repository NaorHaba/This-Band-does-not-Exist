import logging
import os
import random
from functools import partial

from flask import Flask, request, Response, jsonify
from flask_cors import cross_origin
import json
import pandas as pd

from band_maker.band_generator import GenerationInput, BandGenerator, GeneratedBand
from band_maker.custom_generate import decrease_temperature_gradually

from pywebio.platform.remote_access import start_remote_access_service

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

BAND_DATA_FILE = '../../data/generated_data.csv'
BAND_COUNT = 10000
RATING_DATA_FILE = '../../data/rating.json'

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return Response("hello")


@app.route('/rating', methods=['POST'])
@cross_origin()
def update_rating():
    rating_input = json.loads(request.data)
    logger.info(f"received {rating_input} as rating input")
    with open(RATING_DATA_FILE, 'r') as f:
        rating_from_file = json.load(f)
    if rating_input['ratingValue'] is not None:
        rating_from_file['total_rating'] += int(rating_input['ratingValue'])
        rating_from_file['count'] += 1
        rating_from_file['avg_rating'] = rating_from_file['total_rating'] / rating_from_file['count']
        with open(RATING_DATA_FILE, 'w') as f:
            json.dump(rating_from_file, f)

    logger.info(f"current rating info: {rating_from_file}")
    return jsonify(rating_from_file)


@app.route('/sample', methods=['POST'])
@cross_origin()
def sample_band():
    gen_input = GenerationInput(**json.loads(request.data))
    logger.info(f"Received input {gen_input}")
    if gen_input.song_name or gen_input.band_name:
        raise
    elif gen_input.genre:
        all_examples = pd.read_csv(BAND_DATA_FILE)
        example = all_examples[all_examples.genre == gen_input.genre].sample(1).iloc[0]
    else:
        random_sample = random.randint(0, BAND_COUNT - 1)
        example = pd.read_csv(BAND_DATA_FILE, skiprows=range(1, 1 + random_sample), nrows=1).iloc[0]
    res = jsonify(GeneratedBand(**example.to_dict()))
    return res


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
    except RuntimeError as err:
        logger.error(err)
        return Response("The system is overloaded at the moment, please try again later.", status=500)


if __name__ == '__main__':
    band_forward_generator = BandGenerator('../../models/gpt2_forward_model',
                                           blacklist_path='../../data/artists_blacklist.pickle',
                                           genres_path='../../data/genres.pickle'
                                           )
    song_name_forward_generator = BandGenerator('../../models/gpt2_reverse_model',
                                                blacklist_path='../../data/artists_blacklist.pickle',
                                                genres_path='../../data/genres.pickle'
                                                )
    if not os.path.isfile(RATING_DATA_FILE):
        with open(RATING_DATA_FILE, 'w') as f:
            initial_rating = {
                "total_rating": 0,
                "count": 0,
                "avg_rating": 0
            }
            json.dump(initial_rating, f)
    print('Starting server....')
    start_remote_access_service(local_port=5000)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_evalex=False)
    # app.run(debug=False)
