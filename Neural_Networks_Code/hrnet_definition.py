
# Esther Vera Moreno
# FDP: Detección satelital de edificios y sus aplicaciones mediante redes neuronales
#------------------------------------------------------------------------------------

# Definición de la red neuronal RESUNET

import tensorflow as tf
#import tensorflow.keras as keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
from keras import backend as K

### HRNET

# Hyperparameters
BN_MOMENTUM = 0.1
BN_EPSILON = 1e-5
INITIALIZER = 'he_normal'
# Functions to build layers
def conv(x, outsize, kernel_size, strides_=1, padding_='same', activation=None):
  return Conv2D(outsize, 
                kernel_size, 
                strides=strides_, 
                padding=padding_,
                kernel_initializer=INITIALIZER, 
                use_bias=False,
                activation=activation)(x)

def BasicBlock(x, size, downsampe=False):
  residual = x
  out = conv(x, size, 3)
  out = BatchNormalization(epsilon=BN_EPSILON,
                           momentum=BN_MOMENTUM)(out)
  out = Activation('relu')(out)
  out = conv(out, size, 3)
  out = BatchNormalization(epsilon=BN_EPSILON,
                           momentum=BN_MOMENTUM)(out)
  if downsampe:
    residual = conv(x, size, 1, padding_='valid')
    residual = BatchNormalization(epsilon=BN_EPSILON,
                                  momentum=BN_MOMENTUM)(residual)
  out = keras.layers.Add()([out, residual])
  out = Activation('relu')(out)
  return out

def BottleNeckBlock(x, size, downsampe=False):
  residual = x
  out = conv(x, size, 1, padding_='valid')
  out = BatchNormalization(epsilon=BN_EPSILON,
                           momentum=BN_MOMENTUM)(out)
  out = Activation('relu')(out)
  out = conv(out, size, 3)
  out = BatchNormalization(epsilon=BN_EPSILON,
                           momentum=BN_MOMENTUM)(out)
  out = Activation('relu')(out)
  out = conv(out, size * 4, 1, padding_='valid')
  out = BatchNormalization(epsilon=BN_EPSILON,
                           momentum=BN_MOMENTUM)(out)
  if downsampe:
    residual = conv(x, size * 4, 1, padding_='valid')
    residual = BatchNormalization(epsilon=BN_EPSILON,
                                  momentum=BN_MOMENTUM)(residual)
  
  out = keras.layers.Add()([out, residual])
  out = Activation('relu')(out)
  return out

def layer1(x):
  x = BottleNeckBlock(x, 64, downsampe=True)
  x = BottleNeckBlock(x, 64)
  x = BottleNeckBlock(x, 64)
  x = BottleNeckBlock(x, 64)
 
  return x

def transition_layer(x, in_channels, out_channels):
  num_in = len(in_channels)
  num_out = len(out_channels)
  out = []
  for i in range(num_out):
    if i < num_in:
      if in_channels[i] != out_channels[i]:
        residual = conv(x[i], out_channels[i], 3)
        residual = BatchNormalization(epsilon=BN_EPSILON,
                                     momentum=BN_MOMENTUM)(residual)
        residual = Activation('relu')(residual)
        out.append(residual)
      else:
        out.append(x[i])
    else:
      residual = conv(x[-1], out_channels[i], 3, strides_=2)
      residual = BatchNormalization(epsilon=BN_EPSILON,
                                    momentum=BN_MOMENTUM)(residual)
      residual = Activation('relu')(residual)
      out.append(residual)
  return out

def branches(x, block_num, channels):
  out = []
  for i in range(len(channels)):
    residual = x[i]
    for j in range(block_num):
      residual = BasicBlock(residual, channels[i])
    out.append(residual)
  return out

def fuse_layers(x, channels, multi_scale_output=True):
  out = []
  for i in range(len(channels) if multi_scale_output else 1):
    residual = x[i]
    for j in range(len(channels)):
      if j > i:
        y = conv(x[j], channels[i], 1, padding_='valid')
        y = BatchNormalization(epsilon=BN_EPSILON,
                               momentum=BN_MOMENTUM)(y)
        y = keras.layers.UpSampling2D(size=2 ** (j - i))(y)
        residual = keras.layers.Add()([residual, y])
      elif j < i:
        y = x[j]
        for k in range(i - j):
          if k == i - j - 1:
            y = conv(y, channels[i], 3, strides_=2)
            y = BatchNormalization(epsilon=BN_EPSILON,
                                   momentum=BN_MOMENTUM)(y)
          else:
            y = conv(y, channels[j], 3, strides_=2)
            y = BatchNormalization(epsilon=BN_EPSILON,
                                   momentum=BN_MOMENTUM)(y)
            y = Activation('relu')(y)
        residual = keras.layers.Add()([residual, y])
    residual = Activation('relu')(residual)
    out.append(residual)
  
  return out

# Functions to create model
def HighResolutionModule(x, channels, multi_scale_output=True):
  residual = branches(x, 4, channels)
  out = fuse_layers(residual, channels,
                    multi_scale_output=multi_scale_output)
  return out

def stage(x, num_modules, channels, multi_scale_output=True):
  out = x
  for i in range(num_modules):
    if i == num_modules - 1 and multi_scale_output == False:
      out = HighResolutionModule(out, channels,
                                 multi_scale_output=False)
    else:
      out = HighResolutionModule(out, channels)
  return out

def hrnet_keras(input_size=(256, 256, 3)):
  channels_2 = [32, 64]
  channels_3 = [32, 64, 128]
  channels_4 = [32, 64, 128, 256]
  num_modules_2 = 1
  num_modules_3 = 4
  num_modules_4 = 3
  inputs = Input(input_size)
  x = conv(inputs, 64, 3, strides_=2)
  x = BatchNormalization(epsilon=BN_EPSILON,momentum=BN_MOMENTUM)(x)
  x = conv(x, 64, 3, strides_=2)
  x = BatchNormalization(epsilon=BN_EPSILON,momentum=BN_MOMENTUM)(x)
  x = Activation('relu')(x)
  la1 = layer1(x)
  tr1 = transition_layer([la1], [256], channels_2)
  st2 = stage(tr1, num_modules_2, channels_2)
  tr2 = transition_layer(st2, channels_2, channels_3)
  st3 = stage(tr2, num_modules_3, channels_3)
  tr3 = transition_layer(st3, channels_3, channels_4)
  st4 = stage(tr3, num_modules_4, channels_4,
              multi_scale_output=False)
  up1 = keras.layers.UpSampling2D()(st4[0])
  up1 = conv(up1, 32, 3)
  up1 = BatchNormalization(epsilon=BN_EPSILON,
                           momentum=BN_MOMENTUM)(up1)
  up1 = Activation('relu')(up1)
  up2 = keras.layers.UpSampling2D()(up1)
  up2 = conv(up2, 32, 3)
  up2 = BatchNormalization(epsilon=BN_EPSILON,
                           momentum=BN_MOMENTUM)(up2)
  up2 = Activation('relu')(up2)
  final = conv(up2, 1, 1, padding_='valid', activation='sigmoid')
  model = Model(inputs=inputs, outputs=final)
  
  return model