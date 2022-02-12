import time

import requests
from flask import Flask, request, Response, jsonify
from flask_cors import cross_origin
import json


app = Flask(__name__)


@app.route('/bands', methods=['POST'])
@cross_origin()
def generate_band():
    try:
        time.sleep(5)
        print(json.loads(request.data))
        return Response("Welcome to the student polls management service!")
    except RuntimeError:
        return Response("You are already registered!", status=409)


if __name__ == '__main__':
    app.run(debug=True)  # TODO port=config.port
