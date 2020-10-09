#!/usr/bin/env python3

from tensorflow.keras import Input, Add, Dense, Activation, ZeroPadding2D,
        BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D


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
               kernel_initializer='he_normal')(X)
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
               name=f'{convName}_2a')(X)
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
