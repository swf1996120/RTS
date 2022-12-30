#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Q1mi"
# Date: 2022/6/17

from keras import Input, regularizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, \
    GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.utils import np_utils


# Nin
def model_caltech_vgg16(X_train, Y_train, X_test, Y_test, filepath='./model/model_caltech_Nin.hdf5', nb_epoch=20, ):
    ### modify
    print('Train:{},Test:{}'.format(len(X_train), len(X_test)))
    nb_classes = 101
    y_train = np_utils.to_categorical(Y_train, nb_classes)
    y_test = np_utils.to_categorical(Y_test, nb_classes)
    print('data success')
    input_shape = X_train.shape[1:]

    # base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(28, 28, 1))
    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # for layer in model_vgg.layers:  # 冻结权重
    #     layer.trainable = False
    model = Flatten()(model_vgg.output)
    model = Dense(1024, activation='relu', name='fc1')(model)
    # model = Dropout(0.5)(model)
    model = Dense(512, activation='relu', name='fc2')(model)
    # model = Dropout(0.5)(model)
    model = Dense(nb_classes, activation='softmax', name='prediction')(model)
    model = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pretrain')
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', mode='auto',
                                 save_best_only='True')
    model.fit(X_train, y_train, batch_size=32, nb_epoch=nb_epoch, validation_data=(X_test, y_test),
              callbacks=[checkpoint])
    model = load_model(filepath)
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)

# if __name__ == '__main__':
#     model = load_model("../model/model_mnist_LeNet5.hdf5")
#     print(model)