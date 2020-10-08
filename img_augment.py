#!/usr/bin/env python3

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
def augmentImage(ifile, outDir, c):
    '''
    Given a image file name and directory it augments the images for specified
    number of times and saves the resultant augmented images in the given
    output directory
    '''
    # extract the file name and extension
    fname, ext = os.path.basename(ifile).rsplit('.')
    # creating file to generate correct output incase of error
    iname = fname + '.' + ext
    # check if the image is of valid image format
    if ext not in ['jpeg', 'jpg', 'png']:
        print(f'{WARNING}[ERROR] {iname} is not a recognized image file\
              {ENDC}', file=sys.stderr)
        return 1
    # check if the file exists or not
    elif not os.path.exists(ifile):
        print(f'{WARNING}[ERROR] {iname} not found in the specified directory\
              {ENDC}', file=sys.stderr)
        return 1
    # if all the previous check passes then augment the image and save it
    else:
        # expand the image for the model to work with
        # TODO: open the image file, augment the image
        # augment for specified number of times
        for i in range(c):
            # generate new file name to save the image
            newFname = f'{fname}_augmented_{i+1}.{ext}'
            newFullPath = os.path.sep.join([outDir, newFname])
            # augment the image randomly
            # augImg = imageAugmentation(expImg)
            # TODO: save the augmented image file
            print(f'{OKGREEN}[INFO] saving {newFullPath}{ENDC}')


if args.all:
    for img in glob.glob(os.path.sep.join([imgDir, '*.jpeg'])):
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(img, outDir, args.count)
    for img in glob.glob(os.path.sep.join([imgDir, '*.jpg'])):
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(img, outDir, args.count)
    for img in glob.glob(os.path.sep.join([imgDir, '*.png'])):
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(img, outDir, args.count)
else:
    for img in imageList:
        tempImg = os.path.sep.join([imgDir, img])
        if args.verbose:
            print(f'{OKBLUE}[INFO] augmenting {img} {args.count} times...\
                  {ENDC}')
        augmentImage(tempImg, outDir, args.count)
