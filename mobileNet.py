import numpy as np
from keras import backend as K
from keras.models import Sequential,Input,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from keras.applications import MobileNetV2




#Start
train_data_path      = '../Datasets_2/train'
val_data_path        = '../Datasets_2/val'
test_data_path       = '../Datasets_2/test'
img_rows             = 256
img_cols             = 256
epochs               = 50
batch_size           = 8
num_of_train_samples = 378 
num_of_val_samples   = 98
num_of_test_samples  = 115
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




M_conv = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
for layer in M_conv.layers[:]:
    layer.trainable = True
 

model = Model(inputs = M_conv.input, outputs=M_conv.output) 
Flatten = Flatten()(model.output)
Dense_2 = Dense(4096)(Flatten)
Dense_96 = Dense(512 ,activation='sigmoid')(Dense_2)
Dense_5 = Dense(4, activation='softmax')(Dense_96)
model = Model(input=model.input, output=Dense_5)
print(model.summary())



model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

 

#Train
model.fit_generator(train_generator,
                    #steps_per_epoch=num_of_train_samples // batch_size,
                    steps_per_epoch=num_of_train_samples // batch_size,
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
