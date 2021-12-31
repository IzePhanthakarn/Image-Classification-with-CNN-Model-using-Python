import tensorflow
import os
import pandas as pd
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import to_categorical
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import *

class2text = {
    0: 'mouse',
    1: 'gamepad',
    2: 'shoe',
    3: 'watch'
}
dataset_folder = 'dataset_item'
h5_name = 'mydataset_item.h5'
num_classes = len(class2text)
print("Class Name = ", class2text)              # show Class
print("Number of Class = ", num_classes)        # number of Class
print("Data Set Folder = ", dataset_folder)     # show Path folder of Data Set
print("Model .h5 = ", h5_name)                  # show name of Model .h5 

# function for load an image from a data set into a variable
def load_dataset(data_df):
    x = []
    y = []
    for i, r in data_df.iterrows():
        f = os.path.join(dataset_folder, r['filename'])   # name of picture file
        c = r['class']                                    # number of Class (0,1,2)
        img = image.load_img(f, target_size=(224, 224))   # load picture and resize to 224x224
        img = image.img_to_array(img)                     # change picture to array
        x.append(img)                                     # send picture to x variable
        y.append(c)                                       # send class to y variable
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


# load images used for training
print("Loading Training set")
train_df = pd.read_csv(os.path.join(dataset_folder, 'train_dataset.csv'))
raw_train_x, raw_train_y = load_dataset(train_df)
print("*************************")
print("raw_train_x = ", raw_train_x.shape)
print("raw_train_y = ", raw_train_y.shape)
print("*************************")

# load images used for testing
print("Loading Testing set")
test_df = pd.read_csv(os.path.join(dataset_folder, 'test_dataset.csv'))
raw_test_x, raw_test_y = load_dataset(test_df)
print("*************************")
print("raw_test_x = ", raw_test_x.shape)
print("raw_test_y = ", raw_test_y.shape)
print("*************************")

# Preprocess for input array
train_x = preprocess_input(raw_train_x)
test_x = preprocess_input(raw_test_x)

# change number of Class into one-hot format
# such as 0 --> [1,0,0,0] , 1 --> [0,1,0] , 2 --> [0,0,1]
train_y = to_categorical(raw_train_y, num_classes)
test_y = to_categorical(raw_test_y, num_classes)

# change data type images to float32
train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)

# change data type number of Class to int
train_y = train_y.astype(np.int)
test_y = test_y.astype(np.int)

# load MobileNetV2 Model to be Base for pre-train model
# no bring last layer (include_top=False) because don't have the same number of Classes as imagenet.
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False)

# start to build Model for Data Set
model = Sequential()
# Base model layers
model.add(base_model)
# Pooling layer
model.add(GlobalAveragePooling2D())
# Fully-connected layer
model.add(Dense(1024, activation='relu'))
# Softmax layer
model.add(Dense(num_classes, activation='softmax'))

# show al layer of Model
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# Freeze some layers in the MobileNetV2
n_freezes = 82
for layer in base_model.layers[:n_freezes]:
    layer.trainable = False
for layer in base_model.layers[n_freezes:]:
    layer.trainable = True

# set training
epochs = 10
batch_size = 16
learning_rate = 0.01
optimizer = keras.optimizers.SGD(lr=learning_rate)
loss = keras.losses.categorical_crossentropy

# compile model
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

# train model
hist = model.fit(
    train_x, train_y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1)

# show loss and accuracy
score = model.evaluate(x=test_x, y=test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save Model .h5
model.save(dataset_folder+'/'+h5_name)
print("Your model h5 : "+h5_name)
print("----- Step5 Done -----")
