from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import librosa

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')


def features_extractor(audio):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=100, n_mfcc=50)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


# Define how the api will respond to the post requests
class EarthQuakeClassifier(Resource):

    def post(self):
        json_data = request.get_json(force=True)

        X = json_data['data']
        X = features_extractor(np.asarray(X))
        print("El valor de X es " + str(X))

        X_test_1 = np.array([X, ])
        y_pred_1 = model.predict(X_test_1)
        print(y_pred_1)
        values = np.argmax(y_pred_1)
        print(values)
        mensaje = '{"value":' + str(values) + '}'

        return json.loads(mensaje)


api.add_resource(EarthQuakeClassifier, '/predict')

if __name__ == '__main__':
    model = tf.keras.models.load_model('model.h5')

    app.run(debug=True, host="0.0.0.0", port=3000)
