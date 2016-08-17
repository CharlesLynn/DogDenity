from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from datetime import datetime





def fit_image_generators(train_data_dir, validation_data_dir, img_width, img_height):
    # Augmentation configuration used for training
    train_datagen = ImageDataGenerator(rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    # Augmentation configuration used for testing:
    # only normalizing 
    test_datagen = ImageDataGenerator(rescale=1./255)



    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            #class_mode='binary',
            shuffle=True)
    

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            #class_mode='binary',
            shuffle=True)

    classes = train_generator.nb_class
    n_train_samples = train_generator.nb_sample
    n_val_samples = validation_generator.nb_sample

    return train_generator, validation_generator, classes, n_train_samples, n_val_samples


def build_CNN(classes, img_width, img_height):

    model = Sequential()
    model.add(Convolution2D(64, 3, 3, input_shape=(3, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.15))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.15))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(classes))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

   
def train_network(model, n_epochs, n_train_samples, n_val_samples, batch_size):
    
    #Set callbacks:
    callbacks = [ModelCheckpoint('weights/' + str(classes) +'best.h5', monitor='val_loss', \
    verbose=0, save_best_only=True, mode='auto')]

    #Train model
    hist = model.fit_generator(
            train_generator,
            samples_per_epoch=train_generator.nb_sample,
            nb_epoch=n_epoch,
            validation_data=validation_generator,
            nb_val_samples=validation_generator.nb_sample,
            callbacks=callbacks
            )

    return hist



def save_training_history(hist, n_epoch, batch_size, classes, n_train_samples, n_val_samples):
    ## --- Save Settings ---
    datetime_str = str(datetime.now()).split('.')[0]

    #Save Weights & Model
    weights_path = 'weights/' + str(classes) + '.h5'
    architecture_path = 'weights/' + str(classes) + '.json'
    model.save_weights(weights_path, overwrite=True)
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)

    #Save Parameters and Accuracy
    parameters = '\nn_train_samples: {}, n_validation_samples: {}, n_epoch: {}, batch_size: {}\n'.format(n_train_samples, n_val_samples, n_epoch, batch_size)
    accuracy = 'acc: {}, loss: {}, val_acc: {}, val_loss: {}'.format(*hist.history.values())
    text = '\n' + datetime_str + parameters + accuracy
    with open('log.txt', "a") as myfile:
        myfile.write(text)

    print "Saved!"


if __name__ == '__main__':
    #Set Parameters
    img_width, img_height = 150, 150
    train_data_dir = '../data/train'
    validation_data_dir = '../data/test'
    n_epoch = 150
    batch_size= 128

    #fit_image_generators, build CNN, train_network, save history
    train_generator, validation_generator, classes, n_train_samples, n_val_samples = fit_image_generators(train_data_dir, validation_data_dir, img_width, img_height)
    model = build_CNN(classes, img_width, img_height)
    hist = train_network(model, n_epoch, n_train_samples, n_val_samples, batch_size)
    save_training_history(hist, n_epoch, batch_size, classes, n_train_samples, n_val_samples)
    
