import json
import argparse
import os
import time
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from keras.models import load_model
from numpy import argsort

from preproc.preprocessing.embeddings import embed

import logging

logger = logging.getLogger(__name__)


class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.labels_to_index = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.labels_to_index.items()}

    def __eq__(self, target):
        return\
            self.model.__dir__() == target.model.__dir__() and\
            self.params == target.params and\
            self.labels_to_index == target.labels_to_index and\
            self.labels_index_inv == target.labels_index_inv

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # TODO: CODE HERE
        # load model
        model = load_model(f"{artefacts_path}/model.h5")

        # TODO: CODE HERE
        # load params

        params = json.load(open(os.path.join(artefacts_path, 'params.json'), 'rb'))

        # TODO: CODE HERE
        # load labels_to_index
        labels_to_index = json.load(open(os.path.join(artefacts_path, 'labels_index.json'), 'rb'))

        return cls(model, params, labels_to_index)

    def predict(self, text_list, top_k=3):
        """
            predict top_k tags for a list of texts
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict
        """
        tic = time.time()

        logger.info(f"Predicting text_list=`{text_list}`")
        print("testlist",text_list)
        # TODO: CODE HERE
        # embed text_list
        embeddings = embed(text_list)
        print(embeddings)
        # TODO: CODE HERE
        # predict tags indexes from embeddings
        predictions = self.model.predict(embeddings)
        print(predictions)
        # TODO: CODE HERE
        # from tags indexes compute top_k tags for each text
        indices = argsort(predictions)[0][-top_k:]
        
        #list_indices = [index.argmin() for index in indices]
        predictions = [self.labels_to_index.get(str(index)) for index in indices]
        logger.info("Prediction done in {:2f}s".format(time.time() - tic))

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))
