#!/usr/bin/env python3

from PIL import Image
import numpy as np
import argparse
import requests
import base64
import glob
import sys
import os

# some colors
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# set up the command-line usage
ap = argparse.ArgumentParser(description='detect face mask from images')
ap.add_argument('-i', '--img', help='path to the image to be detected')
ap.add_argument('-d', '--dir', help='path to the directory containing images')
ap.add_argument('-o', '--out', default=os.getcwd(),
                help='output directory or file path')
ap.ad_argument('-v', '--verbose', action='store_true',
               help='get a more verbose output')

# add the usage information
if sys.argv == 1:
    ap.print_help(sys.stderr)
    sys.exit(1)

# get the values of the command line arguments
args = ap.parse_args()

# make the list of images if images are specified
if not args.dir:
    imageList = args.img.split()
    outDir = os.getcwd()
else:
    imageList = glob.glob(os.path.sep.join([args.dir, '*']))
    outDir = args.dir

# src directory
srcDir = os.getcwd()

# do some primary checks
for idx, imgs in enumerate(imageList):
    if imgs.rsplit('.')[-1] not in ['jpg', 'jpeg', 'png']:
        print(f'{WARNING}[INFO] {imgs} is not an image file, skipping...{ENDC}')
        imageList.pop(idx)

# get the paths of images if directory not mentioned
if not args.dir:
    for idx, imgs in enumerate(imageList):
        imageList[idx] = os.path.sep.join([srcDir, imgs])

# initialize the image path
data = 'D://face-mask-detection/dataset/val/iwm/00007_Mask_Mouth_Chin.jpg'

# initialize image data list
imgData = []

for imgPath in imageList:
    # open the image
    # get the image bytes
    with open(imgPath, 'rb') as f:
        # encode the image to send it to server
        if args.verbose:
            print(f'{OKBLUE} encoding {imgPath}{ENDC}')
        img = base64.b64encode(f.read()).decode('utf-8')
    # get the file name and extension for later use
    fname, ext = data.split(os.sep)[-1].rsplit('.')
    imgData.append((img, fname, ext))

# prepare the payload
if args.verbose:
    print(f'{OKGREEN}[INFO] preparing the payload{ENDC}')
payload = {'imgB': imgData}

print('sending the request')
# make the POST request
r = requests.post("http://127.0.0.1:5000/detect", data=payload)

# get the data from response
res = r.json()

# get all the images
for (imgList, fname, ext) in res['detected_faces']:
    nparr = np.array(imgList, dtype=np.uint8)
    imgToSave = Image.fromarray(nparr)
    outFname = f'{fname}_detected_face(s).{ext}'
    outPath = os.path.sep.join([outDir, outFname])
    if args.verbose:
        print(f'{OKBLUE}[INFO] saving the image with detected faces in {outPath}{ENDC}')
    imgToSave.save(outPath)

print(f'{OKGREEN} All done!!{ENDC}')
