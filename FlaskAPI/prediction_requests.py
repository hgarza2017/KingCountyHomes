import requests
from testing_data import data_in

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
d_in = data_in.to_json()
data = {'input': d_in}

r = requests.get(URL, headers=headers, json=data)

r.json()