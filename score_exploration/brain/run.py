
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/home/Developer/NCSN-TF2.0/")
DATADIR = "/home/Developer/NCSN-TF2.0/data/"

import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
from IPython.display import display
from matplotlib.pyplot import imshow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import model, data_loader

print(tf.__version__)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.list_physical_devices('GPU')

OOD = True

X_train, X_test, y_train, y_test = data_loader.load()
model = model.build_model(X_train, y_train)

# if OOD:
#     y_train[y_train > 0] = 1
#     y_test[y_test > 0] = 1


# enc = OneHotEncoder(sparse=False).fit(y_train)
# y_train = enc.transform(y_train)
# y_test = enc.transform(y_test)

class_weight = compute_class_weight("balanced", 
                                    classes=np.unique(data["labels"]),
                                    y=data["labels"])
class_weight = {i:w for i,w in enumerate(class_weight)}
print(class_weight)

t = int(time.time())
log_dir = 'logs/{}'.format(t)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

checkpoint_path= os.path.join("./checkpoints", 'run_ood{}_{}'.format(int(OOD), t))
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path += "/{epoch:02d}-{val_loss:.2f}.hdf5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                save_best_only=False)

model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test),
          callbacks=[cp_callback],
          class_weight=class_weight,
          shuffle=True
         )
