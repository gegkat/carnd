import csv
import numpy as np
import cv2
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
import time
from keras.utils import np_utils
import os
import json
from sklearn.model_selection import train_test_split

N_ROWS = 160
N_COLS = 320
N_CHANNELS = 3

def read(d='.'):
  fname = d + '/driving_log.csv'
  imgs = []
  steers = []
  with open(fname, 'r') as f:
    cr = csv.reader(f);
    count = 0
    for row in cr:
      count = count + 1
      if count > 1:
        for j in range(3):
          src = row[j]
          tokens = src.split('/')
          filename = tokens[-1]
          local_path = "IMG/" + filename
          imgs.append(local_path)

        curr = float(row[3])
        correction = 0.2
        steers.append(curr)
        steers.append(curr+correction)
        steers.append(curr-correction)

  return imgs, steers

def genN(N, imgs, steers, d='.'):
  count = 0
  for X,y in gen(imgs, steers, d):
    count += 1
    print('gen {}: X shape {},  yval {}'.format(count, X.shape, y))
    if count >= N:
      break
  return X,y

def gen(imgs, steers, d):
  for img, steer in itertools.cycle(zip(imgs, steers)):
    X = cv2.imread(d+'/'+img)
    X_flip =  cv2.flip(X,1)
    X = X.reshape(1,N_ROWS,N_COLS,N_CHANNELS)
    y = np.array(steer)
    y = y.reshape(-1,1)
    yield X, y

    X = X_flip.reshape(1,N_ROWS,N_COLS,N_CHANNELS)
    y = -1.0*y
    yield X, y

def get_model():

  model = Sequential()
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(N_ROWS,N_COLS,N_CHANNELS)))
  model.add(Cropping2D(cropping=((70,25),(0,0))))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model

def train(d='.'):

  imgs, steers = read(d)
  imgs_train, imgs_val, steers_train, steers_val = train_test_split(imgs, steers, test_size=0.2)

  print('Training size.   imgs: {}, steering: {}'.format(len(imgs_train)*2, len(steers_train)*2))
  print('Validation size. imgs: {}, steering: {}'.format(len(imgs_val)*2,   len(steers_val)*2))
  print('First training data: {}, {}'.format(imgs_train[0], steers_train[0]))
  print('First validation data: {}, {}'.format(imgs_val[0], steers_val[0]))

  print('Sample gen 4 training:')
  genN(4, imgs_train, steers_train, d)
  print('Sample gen 4 validation:')
  genN(4, imgs_val, steers_val, d)

  model = get_model()
  model.summary()
  model.fit_generator(
    gen(imgs_train, steers_train, d),
    samples_per_epoch=38572, #5000,
    nb_epoch=5, #5,
    validation_data=gen(imgs_val, steers_val, d),
    nb_val_samples=9644 #2000
   )

  print("Saving model weights and configuration file.")

  if not os.path.exists("./steering_model"):
      os.makedirs("./steering_model")

  model.save_weights("./steering_model/steering_angle.h5", True)
  with open('./steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

  return model

if __name__ == "__main__":
  train('sample_data')
#  genN(6, 'sample_data')
