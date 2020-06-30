# https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# create a data generator
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# load and iterate training dataset
train_it = train_datagen.flow_from_directory('coronaImages/train/', class_mode='categorical', batch_size=16)

# load and iterate validation dataset
val_it = val_datagen.flow_from_directory('coronaImages/val/', class_mode='categorical', batch_size=16)

# load and iterate test dataset
test_it = test_datagen.flow_from_directory('coronaImages/test/', class_mode='categorical', batch_size=16)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


img_width, img_height = 256, 256
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

epochs = 50
# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# fit model
model.fit_generator(train_it, epochs = epochs, steps_per_epoch=16, validation_data=val_it, validation_steps=8)


# evaluate model
loss = model.evaluate_generator(test_it, steps=24)

# make a prediction
#yhat = model.predict_generator(predict_it, steps=24)

