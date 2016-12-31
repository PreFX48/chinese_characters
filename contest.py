
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import itertools
from skimage import filters
from skimage import morphology
from skimage import exposure
import pandas

print('Loading...')

data = np.load('train.npy')

train_samples = data[:125000, 0]
train_labels = data[:125000, 1]
test_samples = data[125000:, 0]
test_labels = data[125000:, 1]

print('Done!')
print('Preprocessing...')


# In[ ]:

IMG_ROWS = 64
IMG_COLUMNS = 64
mean_image = np.zeros((IMG_ROWS, IMG_COLUMNS), dtype='float32')
for i in range(train_samples.shape[0]):
    img = imresize(train_samples[i], (IMG_ROWS, IMG_COLUMNS), interp='lanczos')
    img = img.astype('float32')
    img /= 255.0
    img = exposure.rescale_intensity(img)
    train_samples[i] = img
    mean_image += img
for i in range(test_samples.shape[0]):
    img = imresize(test_samples[i], (IMG_ROWS, IMG_COLUMNS), interp='lanczos')
    img = img.astype('float32')
    img /= 255.0
    img = exposure.rescale_intensity(img)
    test_samples[i] = img
    mean_image += img

mean_image /= (train_samples.shape[0] + test_samples.shape[0])
for i in range(train_samples.shape[0]):
    train_samples[i] -= mean_image
for i in range(test_samples.shape[0]):
    test_samples[i] -= mean_image


# In[ ]:
print('Done!')
print('Transposing...')

train_samples = np.dstack(train_samples).transpose((2, 0, 1))
test_samples = np.dstack(test_samples).transpose((2, 0, 1))


print('Done!')
# ## CNN

# In[ ]:

minLabel = np.min(train_labels)
maxLabel = np.max(train_labels)
print('range from', minLabel, 'to', maxLabel)
sorted_labels = sorted(train_labels.tolist())
unique_labels = []
for k, g in itertools.groupby(sorted_labels):
    unique_labels.append(k)
print(len(unique_labels), 'unique labels')

labels_to_indices = dict()
indices_to_labels = dict()
for i in range(len(unique_labels)):
    labels_to_indices[unique_labels[i]] = i
    indices_to_labels[i] = unique_labels[i]

new_train_labels = np.vectorize(lambda x: labels_to_indices[x])(train_labels)
new_test_labels = np.vectorize(lambda x: labels_to_indices[x])(test_labels)


# In[ ]:

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras import backend as K

np.random.seed(1337)

if K.image_dim_ordering() == 'th':
    train_samples = train_samples.reshape(train_samples.shape[0], 1, IMG_ROWS, IMG_COLUMNS)
    test_samples = test_samples.reshape(test_samples.shape[0], 1, IMG_ROWS, IMG_COLUMNS)
    input_shape = (1, IMG_ROWS, IMG_COLUMNS)
else:
    train_samples = train_samples.reshape(train_samples.shape[0], IMG_ROWS, IMG_COLUMNS, 1)
    test_samples = test_samples.reshape(test_samples.shape[0], IMG_ROWS, IMG_COLUMNS, 1)
    input_shape = (IMG_ROWS, IMG_COLUMNS, 1)

batch_size = 256
nb_classes = len(unique_labels)
nb_epoch = 30

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(new_train_labels, nb_classes)
Y_test = np_utils.to_categorical(new_test_labels, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])

model.fit(train_samples, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(test_samples, Y_test))
score = model.evaluate(test_samples, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


model.save('configuration8_30_64x64')


# In[ ]:
