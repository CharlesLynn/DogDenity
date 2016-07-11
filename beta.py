from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense



# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = '3classes/train'
validation_data_dir = '3classes/validation'
n_train_samples = 300
n_validation_samples = 80
n_epoch = 150
batch_size=30

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)
        # rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
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


#generator test
# n = 0
# for img in train_generator:
#     n += 1
#     print img[0].shape

#     if n == n_train_samples: break





model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(3))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit_generator(
        train_generator,
        samples_per_epoch=n_train_samples,
        nb_epoch=n_epoch,
        validation_data=validation_generator,
        nb_val_samples=n_validation_samples)


model.save_weights('test.h5', overwrite=True)
print "Done."
