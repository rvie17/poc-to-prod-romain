import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
from predict.predict.run import TextPredictionModel
#model load and cleaning
model = tf.keras.models.load_model("model.h5")

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def get_text():
    #text = request.args.get('text')
    #print(text)
    text =  "this is a python script"
    print(model.summary())
    print(TextPredictionModel.predict("",text))
    #return jsonify(model.predict(text))


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True)