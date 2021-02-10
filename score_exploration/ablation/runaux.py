
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("/home/Developer/NCSN-TF2.0/")

import PIL
import utils, configs
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from helper import plot_confusion_matrix, metrics

from ood_detection_helper import *
from datasets.dataset_loader import  *
from tqdm import tqdm
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from PIL import Image
from IPython.display import display
from matplotlib.pyplot import imshow
from datetime import datetime

import seaborn as sns
import plotly as py
import plotly.graph_objs as go

import umap
from sklearn.preprocessing import MinMaxScaler 

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16

sns.set(style="darkgrid")
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16

# seed=42
# tf.random.set_seed(seed)
# np.random.seed(seed)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))
tf.__version__

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

LABELS = np.array(["Train", "CIFAR", "Outliers"])
COLORS = sns.color_palette("bright")
results = {}

## Hyperprameter analysis

NUM_Ls = [1, 3, 7, 10, 15, 20]
SIGMAs = [0.5,1,2,10]
print(sys.argv)
EXPERIMENT = int(sys.argv[1])
ID = int(sys.argv[2])

if EXPERIMENT == 0: # Run NUM_L experiment
    for L in NUM_Ls:  #NUM_Ls
        fname = "SH1e+00_L{}.p".format(L)
        with open(fname, "rb") as f:
            score_dict = pickle.load(f)

        train_scores = np.array(score_dict["train"])
        inlier_scores = np.array(score_dict["cifar"])
    #     outlier_scores = [np.array(score_dict["svhn"])]
        outliers = [n for n in list(score_dict.keys()) if n not in ["train", "cifar", "svhn", "gaussian", "uniform" ]]
        outlier_scores = np.concatenate([np.array(score_dict[n] )for n in outliers])

        if L == 1:
            train_scores = train_scores.reshape(-1,1)
            inlier_scores = inlier_scores.reshape(-1,1)
            outlier_scores = outlier_scores.reshape(-1,1)

        print("Collected:", outlier_scores.shape)

        # Train Data = L2-norm(Pixel Scores)
        X_train, X_test =  train_scores.copy(), inlier_scores.copy()
        result_dict = auxiliary_model_analysis(X_train, X_test, [outlier_scores],
                                              LABELS, flow_epochs=1000)

        with open("L_models/{}-{}.p".format(L, ID), "wb") as f:
            pickle.dump(result_dict, f)
else:
    for SH in SIGMAs:  
        fname = "SH{:.0e}_L10.p".format(SH)
        with open(fname, "rb") as f:
            score_dict = pickle.load(f)

        train_scores = np.array(score_dict["train"])
        inlier_scores = np.array(score_dict["cifar"])
        outliers = [n for n in list(score_dict.keys()) if n not in ["train", "cifar", "svhn", "gaussian", "uniform" ]]
        outlier_scores = np.concatenate([np.array(score_dict[n] )for n in outliers])
        print("Collected:", outlier_scores.shape)

        # Train Data = L2-norm(Pixel Scores)
        X_train, X_test =  train_scores.copy(), inlier_scores.copy()
        result_dict = auxiliary_model_analysis(X_train, X_test, [outlier_scores],
                                              LABELS, flow_epochs=1000)

        with open("SH_models/{:.0e}-{}.p".format(SH, ID), "wb") as f:
            pickle.dump(result_dict, f)
