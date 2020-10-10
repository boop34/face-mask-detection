#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import argparse
import shutil
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

# setting up parser for command-line options and arguments
ap = argparse.ArgumentParser(description='Creating train, validation, test \
                             sub directories from given source directory')
ap.add_argument('-d', '--dir', required=True, type=str,
                help='source directory path to take data from')
ap.add_argument('-o', '--out', default=os.getcwd(), type=str,
                help='output directory to save the train, validation and \
                test data')
ap.add_argument('-r', '--ratio', default='0.6, 0.2, 0.2',
                help='ratio in which to split the data in form of \
                "train, validation, test" [default=0.6, 0.2, 0.2]')

# setting up command-line usage
if (sys.argv) == 1:
    ap.print_help(sys.stderr)

args = ap.parse_args()

# check if the source directory exists and inilialize it
if os.path.exists(args.dir):
    if args.dir[-1] == os.sep:
        sourceDir = args.dir[:-1]
    else:
        sourceDir = args.dir
else:
    print(f'{FAIL}[INFO]cannot find {args.dir} in your file system{ENDC}',
          file=sys.stderr)
    sys.exit(1)

# check if the output directory exists and iniltialize it
# if directory path given explicitly
if args.out != os.getcwd():
    # check if the directory exists
    if os.path.exists(args.out):
        # check if the directory is empty
        if len(os.listdir(args.out)) > 0:
            print(f'{WARNING}{args.out} is not an empty directory. Do you \
                  want to procced ([Y]es/[N]o)?{ENDC}')
            # if the input is [Y]es
            if input().strip().lower() == 'y' \
               or input().strip().lower() == 'yes':
                outDir = args.out
            # if the input is [N]o
            else:
                print(f'{FAIL}[INFO]aborting{ENDC}', file=sys.stderr)
                sys.exit(1)
        # if the directory is empty
        else:
            outDir = args.out
    # if the directory doesn't exists
    else:
        print(f"{FAIL}[INFO]directory doesn't esxists{ENDC}")
# if directory not given explicitly
else:
    outDir = os.path.sep.join([args.out, 'dataset'])
    print(f'{OKBLUE}[INFO] making a directory named "dataset" in{ENDC}',
          f'{OKBLUE}current directory...{ENDC}')
    # try to make the directory
    try:
        os.mkdir(outDir)
    # if directory already exists
    except FileExistsError:
        print(f'{FAIL}[INFO] {outDir} already exists, try giving{ENDC}',
              f'{FAIL}different name{ENDC}', file=sys.stderr)
        sys.exit(1)

# initialize the train, val, test split ratio
trainR, valR, testR = map(float, args.ratio.split(','))

# check if the ratios add up
if not trainR + valR + testR == 1:
    print(f"{FAIL}[INFO] given ratios doesn't add up{ENDC}", file=sys.stderr)
    sys.exit(1)

# make train, validation, test directories in the output directory
os.mkdir(os.path.sep.join([outDir, 'train']))
os.mkdir(os.path.sep.join([outDir, 'val']))
os.mkdir(os.path.sep.join([outDir, 'test']))

# iterate over the label directories
for label in os.listdir(sourceDir):
    print(f'{OKGREEN}[INFO] looking into {label} directory...{ENDC}')
    # get the full path of the directory
    labelDir = os.path.sep.join([sourceDir, label])

    # list all the file names
    dataList = glob.glob(os.path.sep.join([labelDir, '*']))
    # set the seed to reproduce the results
    np.random.seed(2020)
    # shuffle the list to get a more uniform distribution
    np.random.shuffle(dataList)

    # calculate the splits
    split_1 = int(trainR * len(dataList))
    split_2 = int((trainR + valR) * len(dataList))
    # generate the list of train data file names
    trainList = dataList[:split_1]
    # generate the list of val data file names
    valList = dataList[split_1: split_2]
    # generate the list of test file names
    testList = dataList[split_2:]

    # make paths for train, val, test directories
    trainDir = os.path.sep.join([outDir, 'train', label])
    valDir = os.path.sep.join([outDir, 'val', label])
    testDir = os.path.sep.join([outDir, 'test', label])

    # make the directories
    os.mkdir(trainDir)
    os.mkdir(valDir)
    os.mkdir(testDir)

    # to move training data
    # make a pratial function to help with multithreading
    print(f'{OKBLUE}[INFO] moving data from {labelDir} to {trainDir}...\
          {ENDC}')
    moveData = partial(shutil.move, dst=trainDir)
    with ThreadPoolExecutor() as executor:
        executor.map(moveData, trainList)

    # to move validation data
    # make a pratial function to help with multithreading
    print(f'{OKBLUE}[INFO] moving data from {labelDir} to {valDir}...\
          {ENDC}')
    moveData = partial(shutil.move, dst=valDir)
    with ThreadPoolExecutor() as executor:
        executor.map(moveData, valList)

    # to move test data
    # make a pratial function to help with multithreading
    print(f'{OKBLUE}[INFO] moving data from {labelDir} to {testDir}...\
          {ENDC}')
    moveData = partial(shutil.move, dst=testDir)
    with ThreadPoolExecutor() as executor:
        executor.map(moveData, testList)

    # done
    print(f'{OKGREEN}[INFO] finished moving data from {labelDir}')
