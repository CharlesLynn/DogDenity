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

from datetime import datetime



# dimensions of our images.
img_width, img_height = 150, 150


train_data_dir = 'data/val_split'
validation_data_dir = 'data/val_split'
n_epoch = 1
batch_size= 175
patience = 200 #Earlystop Callback, number of epochs without val_acu increase.


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        #class_mode='binary',
        shuffle=True)
classes = train_generator.nb_class

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        #class_mode='binary',
        shuffle=True)



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

# set callback:
callbacks = [ModelCheckpoint('weights/' + str(classes) +'best.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')]


hist = model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.nb_sample,
        nb_epoch=n_epoch,
        validation_data=validation_generator,
        nb_val_samples=validation_generator.nb_sample,
        callbacks=callbacks
        )






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
parameters = '\nn_train_samples: {}, n_validation_samples: {}, n_epoch: {}, batch_size: {}, Patience\n'.format(train_generator.nb_sample, validation_generator.nb_sample, n_epoch, batch_size, patience)
accuracy = 'acc: {}, loss: {}, val_acc: {}, val_loss: {}'.format(*hist.history.values())
text = '\n' + datetime_str + parameters + accuracy
with open('log.txt', "a") as myfile:
    myfile.write(text)
