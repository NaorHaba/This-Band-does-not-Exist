import time

import requests
from flask import Flask, request, Response, jsonify
from flask_cors import cross_origin
import json

from band_maker.band_generator import GenerationInput

app = Flask(__name__)


@app.route('/bands', methods=['POST'])
@cross_origin()
def generate_band():
    gen_input = GenerationInput(**json.loads(request.data))
    if gen_input.band_name:
        generator = band_forward_generator
    else:
        generator = song_name_forward_generator
    try:
        gen_band = generator.generate_by_input(generator,
                                               **generation_args)
        return Response(gen_band)
    except CertainErrors...:
        # TODO
        return Response("You are already registered!", status=409)


if __name__ == '__main__':
    band_forward_generator = ...
    song_name_forward_generator = ...
    app.run(debug=True)  # TODO port=config.port
