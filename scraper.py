#!/usr/bin/env python3

import requests
import argparse
import glob
from time import sleep
import os

# some colors to make the output pretty
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# setting up parser for command-line options and argumens
ap = argparse.ArgumentParser(description='Scraping images from unsplash.com')
ap.add_argument('-q', '--query', required=True,
                help='query to search images from unsplash.com')
ap.add_argument('-ic', '--imageCount', default=50, type=int,
                help='maximum number of images to be downloaded[default: 50]')
ap.add_argument('-o', '--output', default=os.getcwd(),
                help='path to store the downloaded images')
args = ap.parse_args()

# the query keyword
keyword = '%20'.join(args.query.split())

# number of images to be fetched
imageCounter = args.imageCount

# base url for the searches
url = \
f'https://unsplash.com/napi/search/photos?query={keyword}&per_page=20&page='

# the output directory
if args.output == os.getcwd():
    outputDir = os.path.sep.join([args.output, 'download'])
else:
    outputDir = args.output

# create directory if not exists already
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

# file name
fname = 0

# check if file with same name already exists, if so then increment
while glob.glob(os.path.sep.join([outputDir, f'{str(fname).zfill(8)}.*'])):
    fname += 1

#page count
page = 1

print(f'{HEADER}[INFO] Searching unsplash for "{args.query}"...{ENDC}')
print()

while imageCounter > 0:
    # generating url
    searchUrl = url + str(page)
    # parsing the url content
    htmlData = requests.get(searchUrl)
    htmlData.raise_for_status()
    imageData = htmlData.json()

    # traversing over elements from the response to get the image source urls
    for images in imageData['results']:
        # generating filepath to save the image
        filePath = os.path.sep.join([outputDir,
                                     f'{str(fname).zfill(8)}.jpeg'])
        # debug information
        print(f"{OKBLUE}[INFO] fetching {images['urls']['regular']}{ENDC}")
        try:
            response = requests.get(images['urls']['regular'], timeout=30)

            # saving the image to the determined filename
            print(f'{OKGREEN}[INFO] saving image to {filePath}{ENDC}')
            with open(filePath, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            # check if a timeout has occured
            if isinstance(e, requests.exceptions.Timeout):
                print(f'{FAIL}[FAIL] request timed out{ENDC}')
            else:
                # error while saving the file
                print(f'{FAIL}[FAIL] error saving {filePath}{ENDC}')

            # skip that file if error occured
            print(f'{WARNING}[WARNING] skipping the fie {filepath}{ENDC}')
            continue

        # detrecemting the image counter
        imageCounter -= 1
        # incrementing the filename
        fname += 1

        # break if the counter limit is reached
        if imageCounter == 0:
            break

    page += 1

print(f'{HEADER}[INFO] finished downloading {args.imageCount} images{ENDC}')
