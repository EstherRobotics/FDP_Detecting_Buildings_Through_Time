
# Esther Vera Moreno
# FDP: Detección satelital de edificios y sus aplicaciones mediante redes neuronales
#------------------------------------------------------------------------------------

# Definición de la red neuronal SEGNET

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


def segnet(input_shape):

      # Encoding layer
      img_input = Input(shape= input_shape)
      x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)
      x = BatchNormalization(name='bn1')(x)
      x = Activation('relu')(x)
      x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
      x = BatchNormalization(name='bn2')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)
        
      x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
      x = BatchNormalization(name='bn3')(x)
      x = Activation('relu')(x)
      x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
      x = BatchNormalization(name='bn4')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)

      x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
      x = BatchNormalization(name='bn5')(x)
      x = Activation('relu')(x)
      x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
      x = BatchNormalization(name='bn6')(x)
      x = Activation('relu')(x)
      x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
      x = BatchNormalization(name='bn7')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)
      x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
      x = BatchNormalization(name='bn8')(x)
      x = Activation('relu')(x)
      x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
      x = BatchNormalization(name='bn9')(x)
      x = Activation('relu')(x)
      x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
      x = BatchNormalization(name='bn10')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)

      x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
      x = BatchNormalization(name='bn11')(x)
      x = Activation('relu')(x)
      x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
      x = BatchNormalization(name='bn12')(x)
      x = Activation('relu')(x)
      x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
      x = BatchNormalization(name='bn13')(x)
      x = Activation('relu')(x)
      x = MaxPooling2D()(x)
      x = Dense(1024, activation = 'relu', name='fc1')(x)
      x = Dense(1024, activation = 'relu', name='fc2')(x)
      # Decoding Layer 
      x = UpSampling2D()(x)
      x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
      x = BatchNormalization(name='bn14')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
      x = BatchNormalization(name='bn15')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
      x = BatchNormalization(name='bn16')(x)
      x = Activation('relu')(x)
       
      x = UpSampling2D()(x)
      x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
      x = BatchNormalization(name='bn17')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
      x = BatchNormalization(name='bn18')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
      x = BatchNormalization(name='bn19')(x)
      x = Activation('relu')(x)
      x = UpSampling2D()(x)
      x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
      x = BatchNormalization(name='bn20')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
      x = BatchNormalization(name='bn21')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
      x = BatchNormalization(name='bn22')(x)
      x = Activation('relu')(x)
      x = UpSampling2D()(x)
      x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
      x = BatchNormalization(name='bn23')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
      x = BatchNormalization(name='bn24')(x)
      x = Activation('relu')(x)

      x = UpSampling2D()(x)
      x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
      x = BatchNormalization(name='bn25')(x)
      x = Activation('relu')(x)
      x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv13')(x)
      x = BatchNormalization(name='bn26')(x)
      x = Activation('sigmoid')(x)
      pred = Reshape((valueSize,valueSize,1))(x)
       
      model = Model(inputs=img_input, outputs=pred)

      return model
