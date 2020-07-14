import numpy as np
from keras import backend as K
from keras.models import Sequential,Input,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from keras.applications import VGG16




#Start
train_data_path      = '../Datasets/train'
val_data_path        = '../Datasets/val'
test_data_path       = '../Datasets/test'
img_rows             = 256
img_cols             = 256
epochs               = 100
batch_size           = 8
num_of_train_samples = 675
num_of_val_samples   = 231
num_of_test_samples  = 142
image_size           = 256 

#Image Generator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   #rotation_range=40,
                                   #width_shift_range=0.2,
                                   #height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   #horizontal_flip=True,
                                   fill_mode='nearest')


val_datagen  = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255) 

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    #save_to_dir = "./im_gen_outputs"
                                                    class_mode='categorical')


validation_generator = val_datagen.flow_from_directory(val_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

 
test_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')




vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 

model = Model(inputs = vgg_conv.input, outputs=vgg_conv.output) 
Flatten = Flatten()(model.output)
Dense_2 = Dense(4096)(Flatten)
Dense_96 = Dense(512 ,activation='sigmoid')(Dense_2)
Dense_5 = Dense(2, activation='softmax')(Dense_96)
model = Model(input=model.input, output=Dense_5)





model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

 

#Train
model.fit_generator(train_generator,
                    #steps_per_epoch=num_of_train_samples // batch_size,
                    steps_per_epoch=300 // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_val_samples // batch_size)

 

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator, num_of_test_samples // batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = [ 'covid', 'viral']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
