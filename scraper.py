#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
import requests
import argparse
import glob
import os
import sys

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
ap.add_argument('-ic', '--imgCount', default=50, type=int,
                help='maximum number of images to be downloaded \
                [default: 50]')
ap.add_argument('-o', '--output', default=os.getcwd(),
                help='path to store the downloaded images')
ap.add_argument('-v', '--verbose', help='get a more verbos output',
               action='store_true')
ap.add_argument('-res', '--resolution', default='regular',
                help='specify the resolution of the image [options: full, \
                regular, small, thumb][default: regular]')
ap.add_argument('-oset', '--offset', default=0, type=int,
                help = 'number of images to skip before downloading \
                [default: 0]')

# setting up command line usage
if (sys.argv) == 1:
    ap.print_help(sys.stderr)
    sys.exit(1)

args = ap.parse_args()

# the query keyword
keyword = '%20'.join(args.query.split())

# number of images to be fetched
imageCounter = args.imgCount

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

# initializing fail counter to detect how many images could not be saved
failCount = 0

# check if file with same name already exists, if so then increment
while glob.glob(os.path.sep.join([outputDir, f'{str(fname).zfill(8)}.*'])):
    fname += 1

# initializing page count and offset count depending on the offset value
# as unsplash fetches 20 images per page
if args.offset < 20:
    page = 1
    offsetCount = args.offset
else:
    page = args.offset // 20 + 1
    offsetCount = args.offset % 20

# print if verbose option available
if args.verbose:
    print(f'{HEADER}[INFO] Searching unsplash for "{args.query}"...{ENDC}')

def downloadImage(data):
    '''
    data -> (imageUrl, filePath)
    download image from the given imageUrl and save the image into filePath
    '''
    # unpacking data tuple
    imageUrl, filePath = data

    # trying to fetch the image and save it to the file
    try:
        # getting the response after fetching the url
        response = requests.get(imageUrl, timeout=30)

        # writing the image data into the file name
        with open(filePath, 'wb') as f:
            f.write(response.content)
    # handling exception
    except Exception as e:
        # check if a timeout has occured
        if isinstance(e, requests.exceptions.Timeout):
            print(f'{FAIL}[FAIL] request timed out{ENDC}')
        else:
            # error while saving the file
            print(f'{FAIL}[FAIL] error saving {filePath}{ENDC}')

        # skip that file if error occured
        print(f'{WARNING}[WARNING] skipping the fie {filepath}{ENDC}')

        # unsuccessfull attempt
        return False

    # successfull attempt
    return True


def printDebugInfo(dataList, **kwargs):
    '''
    args -> list of tuples of imageUrl, filePath
    kwargs -> mode -> fetch, save
              err  -> to evaluate if a file was saved successfully
    print debug information like the url being fetched or the filename where
    the image is being saved
    '''
    for idx, (imageUrl, filePath) in enumerate(dataList):
        # printing the fetching information
        if kwargs['mode'] == 'fetch':
            print(f"{OKBLUE}[INFO] fetching {imageUrl}{ENDC}")
        # printing the file info where the image was saved
        # also checing if the image save was successful
        elif kwargs['mode'] == 'save' and kwargs['err'][idx]:
            print(f'{OKGREEN}[INFO] saving image to {filePath}{ENDC}')
        # function not used correctly
        else:
            raise Exception('Use correct parameters')
            # Unsuccessfull
            return False

    # Successfull
    return True


while imageCounter > 0:
    # create a empty list to store the required data of a page
    # for multithreading
    dataList = []

    # generating url
    searchUrl = url + str(page)
    # parsing the url content
    htmlData = requests.get(searchUrl)
    htmlData.raise_for_status()
    imageData = htmlData.json()

    # traversing over elements from the response to get the image source urls
    for images in imageData['results']:
        # skip to adjust the offset count
        if offsetCount > 0:
            offsetCount -= 1
            continue

        # generating filepath to save the image
        filePath = os.path.sep.join([outputDir,
                                     f'{str(fname).zfill(8)}.jpeg'])
        imageUrl = images['urls'][args.resolution]

        # adding the image url and file path to the list for multithreading
        dataList.append((imageUrl, filePath))

        # detrecemting the image counter
        imageCounter -= 1
        # increamenting the file name
        fname += 1

        # check if file with same name already exists, if so then increment
        while glob.glob(os.path.sep.join([outputDir,
                                          f'{str(fname).zfill(8)}.*'])):
            fname += 1

        # break if the counter limit is reached
        if imageCounter == 0:
            break

    # print debug info if verbose option given
    if args.verbose:
        printDebugInfo(dataList, mode='fetch')

    # Using multithreading to maximize IO bound download operation
    result = []
    with ThreadPoolExecutor() as executor:
        result = list(executor.map(downloadImage, dataList))

    # increment the fail counter if any
    failCount += len(dataList) - len(list(filter(None, list(result))))

    # print saved the image to the determined filename if verbose on
    if args.verbose:
        printDebugInfo(dataList, mode='save', err=list(result))

    # increamenting the page value to get the contents of the next page
    page += 1

# print if verbose option available
if args.verbose:
    print(f'{HEADER}[INFO] finished downloading images{ENDC}')
    print(f'{OKGREEN}Successfull: {args.imgCount - failCount}{ENDC}')
    print(f'{FAIL}Unsuccessfull: {failCount}{ENDC}')
