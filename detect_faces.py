#!/usr/bin/env python3

import requests
import base64
import json
import os

# initialize the image path
data = '/home/rahi/face mask detection/download/00000000.jpeg'
# open the image
# get the image bytes
with open(data, 'rb') as f:
    img = base64.b64encode(f.read()).decode('utf-8')
# get the file name and extension for later use
fname, ext = data.split(os.sep)[-1].rsplit('.')

# prepare the payload
payload = {'imgB': img, 'ext': ext}
# make the POST request
r = requests.post("http://127.0.0.1:5000/detect", data=payload)

print('sending the request')

# get the data from response
res = r.json()
# get the image bytes
responseBytes = base64.b64encode(res['detected_faces'].encode('utf-8'))

