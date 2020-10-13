#!/usr/bin/env python3

from PIL import Image
import numpy as np
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

print('sending the request')
# make the POST request
r = requests.post("http://127.0.0.1:5000/detect", data=payload)


# get the data from response
res = r.json()
nparr = res['detected_faces']
nparr = np.array(nparr, dtype=np.uint8)
imgToSave = Image.fromarray(nparr)
outFname = f'{fname}_detected_face(s).{ext}'
imgToSave.save(outFname)
