import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from predict.predict.run import TextPredictionModel
#model load and cleaning
#model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def get_text():
    model = TextPredictionModel.from_artefacts("/Users/rvie/Documents/5A/Poc to prod/poc-to-prod-romain/train/data/artefacts/2023-01-03-12-42-59")

    text = "this is a python script"

    predictions = model.predict(text)
    #print(predictions)
    return text


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True)