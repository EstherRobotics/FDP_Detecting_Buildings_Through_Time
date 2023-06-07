
# Esther Vera Moreno
# FDP: Detección satelital de edificios y sus aplicaciones mediante redes neuronales
#------------------------------------------------------------------------------------

# Código de entrenamiento general 

# Importar librerías 
import os
import random
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import resunet_definition
import hrnet_definition
import segnet_definition
import metrics_losses


# Seleccionar red neuronal a entrenar
neural_network = 'RESUNET'
#neural_network = 'HRNET'
#neural_network = 'SEGNET'


# Seleccionar tamaños a entrenar 
sizes = [128]
#sizes = [256]
#sizes = [256,128]


# Añadir callbacks al entrenamiento para parar si la pérdida sube y ajustar el LR
callbacks = [EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)]

# En el entrenamiento/validación solo se utiliza un mes para cada imagen
# Se elige el mes más reciente contenido en la carpeta, por lo que se comprueban en el siguiente orden 
months = ["2020_01", "2019_12", "2019_11", "2019_10", "2019_09"]


# Entrenamiento para los tamaños de chips deseados (128, 256 o ambos)
for valueSize in sizes:

  # Tamaño de las imágenes 
  im_width = valueSize
  im_height = valueSize
  image_size = valueSize
  border = 5


  # Ruta al fichero de chips del tamaño correspondiente y guardar contenido en folders_content
  folders_path = "my_chips_good_masked/my_chips_good_masked/my_chips_" + str(valueSize) + "_masked/"
  folders_content = os.listdir(folders_path)

  # Rutas de las carpetas
  folder_chips = '/chips_' + str(valueSize) + '/' 
  folder_masks = '/masks_' + str(valueSize) + '/'


  # Estas variables sirven para poder hacer una división posterior por imágenes completas
  # Los chips de imágenes para el mes elegido debe usarse completamente o para entrenamiento o para validación o para test 
  # No se deben mezclar chips del mismo lugar en ambos conjuntos para no hacer sobreentrenamiento ya que estos son muy parecidos entre sí a lo largo del tiempo
  test_imgs = 0
  val1 = 0


  # Variables auxiliares para guardar y ordenar las rutas
  X_to_sort = []
  y_to_sort = []
  X_aux = []
  y_aux = []

 


  # Para cada imagen de cada lugar, se selecciona la ruta con los chips tomados del mes más reciente.
  for i, folder in tqdm(enumerate(folders_content), total=len(folders_content)):

    # Se elige el mes a utilizar más reciente 
    for m in months:
        if os.path.isdir(aux_path_chips) and os.path.isdir(aux_path_masks):
            break
        aux_path_chips = folders_path + folder + folder_chips + m
        aux_path_masks = folders_path + folder + folder_masks + m


    # Contenido de las rutas de chips y mask 
    content_chips = os.listdir(aux_path_chips)
    content_mask = os.listdir(aux_path_masks)

    # Esto se hace para dividir después en entrenamiento, validación y test por carpetas y guardar las rutas a cada chip y mask 
    # Se añade el número total de chips dentro de la carpeta del mes seleccionado  
    # Si la carpeta actual es la 41-47, se suma el número de chips contenidos a val1. Estas carpetas se usarán para validación. 
    # Si la carpeta actual es >47, se suma a test_imgs. Estas no se usan en entrenamiento ni validación, se usarán para test 
    for j, img in enumerate(content_chips):
      if(i>40 and i<=47):
        val1 = val1+1

      if(i > 47):
        test_imgs = test_imgs + 1

      # Guardar ruta de cada chip y mask 
      one_path_chip = aux_path_chips + '/' + img
      one_path_mask = aux_path_masks + '/' + img

      X_to_sort.append(one_path_chip)
      y_to_sort.append(one_path_mask)


  # Ordenar rutas de las imágenes y sus máscaras
  X_to_sort.sort()
  y_to_sort.sort()

  # Mostrar número de imágenes de entrenamiento
  print("No. of images = ", len(X_to_sort))

  # Arrays para guardar las imágenes y sus máscaras correspondientes
  X = np.zeros((len(X_to_sort), im_height, im_width, 3), dtype=np.float32)
  y = np.zeros((len(X_to_sort), im_height, im_width, 1), dtype=np.float32)


  # Cargar imágenes y máscaras en arrays, normalizar y guardar en X,y
  for n, id_ in tqdm(enumerate(X_to_sort), total=len(X_to_sort)):
     
      img = img_to_array(load_img(id_))
      mask = img_to_array(load_img(y_to_sort[n], color_mode = "grayscale"))
      
      X[n] = x_img/255.0
      y[n] = mask/255.0



  # Obtener X_trainval que será utilizado para dividir por carpetas el entrenamiento y la validación
  # No se utilizan las imágenes de las carpetas de test (se puede ver que se usa test_imgs y shuffle=False)
  X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_imgs, shuffle = False)
  # Número de imágenes totales para el entrenamiento y validación
  print(len(X_trainval))
  print(len(y_trainval))

  # Se separa entrenamiento y validación por carpetas teniendo en cuenta la cantidad de chips de cada imagen 
  # Se utiliza la variable val1, calculada anteriormente
  x_val = X_trainval[-val1:]
  y_val = y_trainval[-val1:]
  X_train = X_trainval[:-val1]
  y_train = y_trainval[:-val1]

  # Cantidad de imágenes de entrenamiento y validación respectivamente
  print(len(X_train))
  print(len(y_train))
  print(len(x_val))
  print(len(y_val))


  # Seleccionar red neuronal para el enternamiento 
  if(neural_network=='RESUNET'):
    model = ResUNet()

  elif(neural_network=='HRNET'): 
    model = hrnet_keras(input_size=(valueSize, valueSize, 3))

  elif(neural_network=='SEGNET'): 
    model = segnet((valueSize,valueSize,3))


  

  # Compilar modelo con optimizador, pérdida deseada y métrica 
  model.compile(optimizer='Adam', 
                loss = bce_dice_loss, 
                metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), f1_score, precision_m, recall_m, dice_loss])  


  
  # Entrenar red neuronal
  results = model.fit(X_train, y_train, 
  							     batch_size=12, 
  							     epochs=60, 
  							     callbacks=callbacks,
  							     validation_data=(x_val, y_val))



  # Guardar modelo entrenado
  model.save('model_'+neural_network+'_60epochs_' + str(valueSize) + 'M_12bat_black_c_bcedicedefvales.h5')

  # Guardar historial de entrenamiento en csv
  hist_df = pd.DataFrame(results.history) 
  hist_csv_file = 'history_'+neural_network+'_60epochs_'+str(valueSize) + 'M_12bat_black_c_bcedicedefvales.csv'
  with open(hist_csv_file, mode='w') as f:
      hist_df.to_csv(f)


  print("Training finished")
