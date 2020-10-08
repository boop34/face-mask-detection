#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import argparse
import glob
import sys
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

# setting up parser for command line options and arguments
ap = argparse.ArgumentParser(description='Augment images')
ap.add_argument('-i', '--img',
                help='specific image file name(s) from given directory \
                to be augmented')
ap.add_argument('-a', '--all', action='store_true',
                help='all the images from given directory to be augmented')
ap.add_argument('-d', '--dir', default=os.getcwd(),
                help='specify the directory of the images [default: curent \
                directory]')
ap.add_argument('-o', '--output', default=os.getcwd(),
                help='path to store the augmented images [default: current \
                directory]')
ap.add_argument('-v', '--verbose', action='store_true',
                help='get a more verbose output')
ap.add_argument('-c', '--count', default=4, type=int,
                help='the number of images you want to generate [default: 4]')

# setting up usage
if (sys.argv) == 1:
    ap.print_help(sys.stderr)
    sys.exit(1)

args = ap.parse_args()

# initializing the image directory
imgDir = args.dir

# initializing the output directory
outDir = args.output

# if output directory doesn't exist already make one
if not os.path.exists(outDir):
    os.mkdir(outDir)

# make lsit of image names
if not args.all:
    imageList = args.img.split()

# function to augment a given image
def augmentImage(ifile, outDir, c, v):
    '''
    Given a image file name and directory it augments the images for specified
    number of times and saves the resultant augmented images in the given
    output directory
    '''
    # extract the file name and extension
    fname, ext = os.path.basename(ifile).rsplit('.')
    # creating file to generate correct output incase of error
    iname = f'{fname}.{ext}'
    # check if the image is of valid image format
    if ext not in ['jpeg', 'jpg', 'png']:
        print(f'{WARNING}[ERROR] {iname} is not a recognized image file\
              {ENDC}', file=sys.stderr)
        # unsuccessful attempt
        return 1
    # check if the file exists or not
    elif not os.path.exists(ifile):
        print(f'{WARNING}[ERROR] {iname} not found in the specified directory\
              {ENDC}', file=sys.stderr)
        # unsuccessful attempt
        return 1
    # if all the previous check passes then augment the image and save it
    else:
        # open image
        img = Image.open(ifile)
        # make the image a numpy array to do computation
        iarr = np.array(img, dtype='uint8')
        # expand dimension to process it
        iarr = np.expand_dims(iarr, 0)
        # initialize the data augmentor
        imageAugmentation = ImageDataGenerator(rotation_range=35,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              zoom_range=0.3,
                                              fill_mode='nearest')
        # prepare the iterator
        it = imageAugmentation.flow(iarr, batch_size=1)
        # augment for specified number of times
        for i in range(c):
            # generate new file name to save the image
            newFname = f'{fname}_augmented_{i+1}.{ext}'
            newFullPath = os.path.sep.join([outDir, newFname])
            # get the data from the returned tuple of ImageDataGenerator
            data = it.next()[0]
            # convert data to unit8 to make it an image
            augImg = data.astype('uint8')
            # create an Image object from the array
            saveImg = Image.fromarray(augImg)
            # print debug info
            if v:
                print(f'{OKGREEN}[INFO] saving {newFullPath}{ENDC}')
            saveImg.save(newFullPath)

        # successful attempt
        return 0

# for all the images
if args.all:
    # for jpeg images
    for img in glob.glob(os.path.sep.join([imgDir, '*.jpeg'])):
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(img, outDir, args.count,args.verbose)
    # for jpg images
    for img in glob.glob(os.path.sep.join([imgDir, '*.jpg'])):
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(img, outDir, args.count,args.verbose)
    # for png images
    for img in glob.glob(os.path.sep.join([imgDir, '*.png'])):
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(img, outDir, args.count,args.verbose)
# for specific images
else:
    for img in imageList:
        # making the full path string
        imgPath = os.path.sep.join([imgDir, img])
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(imgPath, outDir, args.count,args.verbose)
