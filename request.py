import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'state':2, 'district':9, 'n':6,'p':6,'k':6,'t':6})

print(r.json())