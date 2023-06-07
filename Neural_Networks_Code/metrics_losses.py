  
# Esther Vera Moreno
# FDP: Detección satelital de edificios y sus aplicaciones mediante redes neuronales
#------------------------------------------------------------------------------

# Definición de pérdidas y métricas para el entrenamiento 


import keras
from keras import backend as K


def weighted_cross_entropy(beta):
  def loss(y_true, y_pred):
     weight_a = beta * tf.cast(y_true, tf.float32)
     weight_b = 1 - tf.cast(y_true, tf.float32)
     o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
     return tf.reduce_mean(o)
  return loss


def dice_coeff(y_true, y_pred):
   smooth = 1.
   y_true_f = tf.reshape(y_true, [-1])
   y_pred_f = tf.reshape(y_pred, [-1])
   intersection = tf.reduce_sum(y_true_f * y_pred_f)
   score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
   return score


def dice_loss(y_true, y_pred):
   return 1-dice_coeff(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
   loss = tf.python.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
   return loss


def recall_m(y_true, y_pred):
   y_true = K.ones_like(y_true) 
   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
   all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

   recall = true_positives / (all_positives + K.epsilon())
   return recall

def precision_m(y_true, y_pred):
   y_true = K.ones_like(y_true) 
   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

   predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
   precision = true_positives / (predicted_positives + K.epsilon())
   return precision

def f1_score(y_true, y_pred):
   precision = precision_m(y_true, y_pred)
   recall = recall_m(y_true, y_pred)
   return 2*((precision*recall)/(precision+recall+K.epsilon()))
