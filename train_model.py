#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from ResNet50 import ResNet50
import numpy as np
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

# define where the directory with all the data is
dataDir = os.path.sep.join([os.getcwd(), 'dataset'])

# define categories
# cwm -> correctly wearing mask
# iwm -> incorrectly wearing mask
# nwm -> not wearing mask
categories = ['cwm', 'iwm', 'nwm']


class CustomDataGen(Sequence):
    # initialization
    def __init__(self, imageFiles, labels, batch_size=32,
                 dim=(224, 224, 3), n_classes=3, shuffle=True):
        self.imgFiles = imageFiles
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # the number of batches per epoch
        return int(np.floor(len(self.imgFiles) / self.batch_size))

    def __getitem__(self, index):
        # generate one batch of data
        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]

        # get the image files
        imgFilesBatch = [self.imgFiles[i] for i in indexes]

        # generate data
        X, y = self.__data_generation(imgFilesBatch)

        # return the data
        return X, y

    def on_epoch_end(self):
        # update indexes after each epoch
        self.indexes = np.arange(len(self.imgFiles))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, imgFilesBatch):
        # initialize
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # generate data
        for i, iFile in enumerate(imgFilesBatch):
            # store the data
            try:
                # load the image
                # print(f'{OKBLUE}loading up {iFile}{ENDC}')
                iarr = load_img(iFile, target_size=(224, 224))
                # convert it to numpy array
                iarr = img_to_array(iarr)
                # preprocess the input
                # print(f'{OKBLUE}preprocessing {iFile} to add it to the data\
                # {ENDC}')
                iarr = iarr / 255.0
                iarr = np.expand_dims(iarr, axis=0)
                # print(f'{OKGREEN}{iFile} was sucessfully added to the data\
                # {ENDC}')
            # handling exception
            except ImportError:
                print(f'{WARNING}check if the PIL module is installed\
                        {ENDC}', file=sys.stderr)
            except Exception as e:
                print(e)
                print(f'{FAIL}could not load or process {iFile}{ENDC}',
                      file=sys.stderr)
            # store data
            X[i, ] = iarr

            # store label
            # iFile -> /home/user/face mask detection/dataset/train/cwm/img
            #       -> /home/user/face mask detection/dataset/train/iwm/img
            #       -> /home/user/face mask detection/dataset/train/nwm/img
            # extract the label from the file path
            label = iFile.split(os.path.sep)[-2]
            y[i] = self.labels.index(label)

        # one hot encode the labels
        y = np.eye(self.n_classes)[y.reshape(-1)]
        # return the generated data
        return X, y


# initialize the train and validation directories
trainDir = os.path.sep.join([dataDir, 'train'])
valDir = os.path.sep.join([dataDir, 'val'])

# initialize train and validation file names
trainList = []
valList = []

# load up the train and validation data
for srcDir in [trainDir, valDir]:
    print(f'{OKGREEN}[INFO] loading up {srcDir} names...{ENDC}')
    # look in each directory
    for d in os.listdir(srcDir):
        # set up the directory path for classes
        classDir = os.path.sep.join([trainDir, d])
        # add the files into train file names list
        for filePath in glob.glob(os.path.sep.join([classDir, '*'])):
            print(f'{OKBLUE}[INFO] storing {filePath}..{ENDC}')
            if srcDir == trainDir:
                trainList.append(filePath)
            elif srcDir == valDir:
                valList.append(filePath)
    print(f'{OKGREEN}[INFO] sucessfully loaded {srcDir} names...{ENDC}')

# set up the ResNet50 model
model = ResNet50(input_shape=(224, 224, 3), classes=3)

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# prepare the training and validation generators
trainGen = CustomDataGen(trainList, labels=categories, batch_size=24,
                         n_classes=3, dim=(224, 224, 3), shuffle=True)
valGen = CustomDataGen(valList, labels=categories, batch_size=32,
                       n_classes=3, dim=(224, 224, 3), shuffle=True)

# set up the checkpoint to save model weights during training
cp_callback = ModelCheckpoint(filepath='model/cp.ckpt', save_weights_only=True,
                              verbose=1)

# train the model on the dataset
# set the epoch
epochs = 2
history = model.fit(trainGen, verbose=1, epochs=epochs, validation_data=valGen,
                    callbacks=[cp_callback])

# saving the trained model
print(f'{OKGREEN}[INFO] saving the trained model...{ENDC}')
model.save('model.h5')

# ploting the loss vs accuracy graph
# get the accuracy data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# get the loss data
loss = history.history['loss']
val_loss = history.history['val_loss']

# initialize number of epochs
epochs_range = range(epochs)

# plot the loss graph
plt.figure()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
# save the graph
print(f'{OKBLUE}[INFO] saving the training and validation loss graph{ENDC}')
plt.savefig('loss.png')

# plot the accuracy graph
plt.figure()
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend(loc='lower right')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
# save the graph
print(f'{OKBLUE}[INFO] saving the training and validation accuracy graph{ENDC}')
plt.savefig('accuracy.png')
