#!/usr/bin/env python3

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model


# define identity block
def identityBlock(inputTensor, kernel_size, filters, stage, block):
    # define name to distinguish different blocks
    convName = f'conv{stage}{block}_branch'
    bnName = f'bn{stage}{block}_branch'

    # unpacking filters
    f1, f2, f3 = filters

    # save the input for later use
    X_id = inputTensor

    # block 1
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', name=f'{convName}_2a',
               kernel_initializer='he_normal')(inputTensor)
    X = BatchNormalization(axis=3, name=f'{bnName}_2a')(X)
    X = Activation('relu')(X)

    # block 2
    X = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1),
               padding='same', name=f'{convName}_2b',
               kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name=f'{bnName}_2b')(X)
    X = Activation('relu')(X)

    # block 3
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', name=f'{convName}_2c',
               kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name=f'{bnName}_2c')(X)

    # finally add the input with the output and activate it
    X = Add()([X_id, X])
    X = Activation('relu')(X)

    # return the final output
    return X

# define convolution block
def convBlock(inputTensor, kernel_size, filters, stage, block, strides):
    # define name to distinguish different blocks
    convName = f'conv{stage}{block}_branch'
    bnName = f'bn{stage}{block}_branch'

    # unpacking filters
    f1, f2, f3 = filters

    # save the input for later use
    X_id = inputTensor

    # block 1
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides,
               padding='valid', kernel_initializer='he_normal',
               name=f'{convName}_2a')(inputTensor)
    X = BatchNormalization(axis=3, name=f'{bnName}_2a')(X)
    X = Activation('relu')(X)

    # block 2
    X = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1),
               padding='same', kernel_initializer='he_normal',
               name=f'{convName}_2b')(X)
    X = BatchNormalization(axis=3, name=f'{bnName}_2b')(X)
    X = Activation('relu')(X)

    # block 3
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', kernel_initializer='he_normal',
               name=f'{convName}_2c')(X)
    X = BatchNormalization(axis=3, name=f'{bnName}_2c')(X)

    # processing the input
    X_id = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides,
                  padding='valid', name=f'{convName}_1',
                 kernel_initializer='he_normal')(X_id)
    X_id = BatchNormalization(axis=3, name=f'{bnName}_1')(X_id)

    # finally add the input with the output and activate it
    X = Add()([X_id, X])
    X = Activation('relu')(X)

    # return the final output
    return X

# ResNet50
def ResNet50(input_shape, classes):
    # making a tensor of shape input_shape
    X_input = Input(input_shape)
    # zero padding the tensor
    X = ZeroPadding2D((3, 3))(X_input)

    # stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1',
               kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), name='max_pool')(X)

    # stage 2
    X = convBlock(inputTensor=X, kernel_size=3, filters=[64, 64, 256],
                  stage=2, block='a', strides=(1, 1))
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[64, 64, 256],
                      stage=2, block='b')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[64, 64, 256],
                      stage=2, block='c')

    # stage 3
    X = convBlock(inputTensor=X, kernel_size=3, filters=[128, 128, 512],
                  stage=3, block='a', strides=(2, 2))
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[128, 128, 512],
                      stage=3, block='b')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[128, 128, 512],
                      stage=3, block='c')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[128, 128, 512],
                      stage=3, block='d')

    # stage 4
    X = convBlock(inputTensor=X, kernel_size=3, filters=[256, 256, 1024],
                  stage=4, block='a', strides=(2, 2))
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[256, 256, 1024],
                      stage=4, block='b')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[256, 256, 1024],
                      stage=4, block='c')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[256, 256, 1024],
                      stage=4, block='d')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[256, 256, 1024],
                      stage=4, block='e')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[256, 256, 1024],
                      stage=4, block='f')

    # stage 5
    X = convBlock(inputTensor=X, kernel_size=3, filters=[512, 512, 2048],
                  stage=5, block='a', strides=(2, 2))
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[512, 512, 2048],
                      stage=5, block='b')
    X = identityBlock(inputTensor=X, kernel_size=3, filters=[512, 512, 2048],
                      stage=5, block='c')

    # pooling (average)
    X = AveragePooling2D((7, 7), name='avg_pool')(X)

    # flatten the output layer
    X = Flatten()(X)
    # creating a fully connected layer for predicting classes
    X = Dense(classes, activation='softmax', name=f'fc_{classes}',
              kernel_initializer='he_normal')(X)

    # finally creating a model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    # return the model
    return model
