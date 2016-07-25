from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import cv2
import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict

def model_architecture(img_size, n_classes):
    img_width, img_height = img_size

    #Declare Model Architecture
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=(3, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def network_gridsearch(model, train_data_dir, validation_data_dir, img_size, n_epoch, augment, batch_size=32):
    # dimensions of our images.
    img_width, img_height = img_size
    # train_data_dir = 'data/train_split'
    validation_data_dir = validation_data_dir
    batch_size= 175


    if augment == True:
        train_datagen = ImageDataGenerator(rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)


    # Only rescaling for validation test generator.
    test_datagen = ImageDataGenerator(rescale=1./255)


    #Point train_datagen to image folder and apply Settings.
    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle=True)

    #Point test_datagen to image folder and apply Settings.
    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle=True)

    #Pulls number of classes from the generator.
    n_classes = train_generator.nb_class
    n_images = train_generator.nb_sample

    for epoch in range(1, n_epoch+1):
        print "\nEpoch: {}, TrainDir: {}, Augment: {}".format(epoch, train_data_dir, augment)

        model_hist = model.fit_generator(
                train_generator,
                samples_per_epoch=n_images,
                nb_epoch=1, #One epoch per loop.
                validation_data=validation_generator,
                nb_val_samples=validation_generator.nb_sample,
                )

        save_network(model, model_hist, epoch, n_images, augment, batch_size)



def save_network(model, model_hist, epoch, n_images, augment, batch_size):
    ## --- Save Settings ---
    datetime_str = str(datetime.now()).split('.')[0]
    n_images = int(round(n_images, -2))
    if augment == True: augment = 1
    else: augment = 0

    #Save Weights & Model
    weights_path = 'gridsearch/' + 'images-{}_agument-{}_epoch-{}.h5'.format(n_images, augment, epoch)
    architecture_path = 'gridsearch/' + 'images-{}_agument-{}_epoch-{}.json'.format(n_images, augment, epoch)
    if not os.path.isdir("gridsearch/"): os.makedirs('gridsearch/')
    model.save_weights(weights_path, overwrite=True)
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)


    ID = 'A{} I{}'.format(augment, n_images)

    if os.path.isfile('gridsearch/gridsearch.json'):
        with open('gridsearch/gridsearch.json') as grid_file:
            grid_dict = json.load(grid_file)
            if ID not in grid_dict: grid_dict[ID] = defaultdict(list)
    else:
        grid_dict = {}
        grid_dict[ID] = defaultdict(list)

    hist = model_hist.history
    grid_dict[ID]['epochs'].append(epoch)
    grid_dict[ID]['acc'].append(hist['acc'][-1])
    grid_dict[ID]['val_acc'].append(hist['val_acc'][-1])
    grid_dict[ID]['loss'].append(hist['acc'][-1])
    grid_dict[ID]['val_loss'].append(hist['val_acc'][-1])

    #Save Data Jason
    with open('gridsearch/gridsearch.json', 'w') as outfile:
        json.dump(grid_dict, outfile)

    print grid_dict

if __name__ == '__main__':
    img_size = (150, 150)
    n_classes = 36
    max_epoch = 175
    agument = True
    model = model_architecture(img_size, n_classes)
    #network_gridsearch(train_data_dir, n_epoch, augment)
    for train_folder in ['data/train_split']:
        network_gridsearch(model, train_folder, 'data/val_split', img_size, max_epoch, agument)

    agument = False
    for train_folder in ['data/train_split']:
        network_gridsearch(model, train_folder, 'data/val_split', img_size, max_epoch, agument)
    print "Done"