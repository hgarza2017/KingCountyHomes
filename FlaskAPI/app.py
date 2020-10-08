#import flask
from flask import Flask, jsonify, request
import json
from testing_data import data_in
import numpy as np
import pandas as pd
import pickle


def load_models():
 file_name = 'models/model_file.p'
 with open(file_name, 'rb') as pickled:
  #data = pickle.load(pickled)
  data = pickle.load(pickled)
  model = data['model']
 return model


app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict():
 # stub input features
 request_json = request.get_json()
 x = request_json['input']
 #x_df = pd.DataFrame(x)
 #x_df.columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']
 #x_in = x_df
 # load model
 #print("before")
 model = load_models()
 prediction = model.predict(data_in)
 p = prediction.tolist()
 response = json.dumps({'Price Prediction': p})
 #response = json.dumps({'response': 'yahhhh!'})
 return response, 200

if __name__ == '__main__':
 application.run(debug=True)