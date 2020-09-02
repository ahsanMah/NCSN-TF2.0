import utils, configs
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_command_line_args(_args):
    parser = utils._build_parser()

    parser = parser.parse_args(_args)

    utils.check_args_validity(parser)

    print("=" * 20 + "\nParameters: \n")
    for key in parser.__dict__:
        print(key + ': ' + str(parser.__dict__[key]))
    print("=" * 20 + "\n")
    return parser

configs.config_values = get_command_line_args([])
SIGMAS = utils.get_sigma_levels().numpy()

@tf.function(experimental_compile=True)
def reduce_norm(x):
    return tf.norm(tf.reshape(x, shape=(x.shape[0], -1)),
                   axis=1, ord="euclidean", keepdims=True)

# Takes a norm of the weighted sum of tensors
@tf.function(experimental_compile=True)
def weighted_sum(x):
    x = tf.add_n([x[i] * s for i, s in enumerate(SIGMAS)])
    return reduce_norm(x, axis=[1,2], ord="euclidean")

@tf.function(experimental_compile=True)
def weighted_norm(x):
    x = tf.concat([reduce_norm(x[i] * s) for i, s in enumerate(SIGMAS)], axis=1)
    return x



def load_model(inlier_name="cifar10", checkpoint=-1, save_path="saved_models/",
                filters=128, batch_size=1000):
    args = get_command_line_args([
        "--checkpoint_dir=" + save_path,
        "--filters=" + str(filters),
        "--dataset=" + inlier_name,
        "--sigma_low=0.01",
        "--sigma_high=1",
        "--resume_from=" + str(checkpoint),
        "--batch_size=" + str(batch_size)
        ])
    configs.config_values = args

    sigmas = utils.get_sigma_levels().numpy()
    save_dir, complete_model_name = utils.get_savemodel_dir() # "longleaf_models/baseline64_fashion_mnist_SL0.001", ""
    model, optimizer, step, _, _ = utils.try_load_model(save_dir,
                                                step_ckpt=configs.config_values.resume_from,
                                                verbose=True)
    return model

def compute_scores(model, x_test):
    
    # Sigma Idx -> Score
    score_dict = []
    
    sigmas = utils.get_sigma_levels().numpy()
    final_logits = 0 #tf.zeros(logits_shape)
    progress_bar = tqdm(sigmas)
    for idx, sigma in enumerate(progress_bar):
        
        progress_bar.set_description("Sigma: {:.4f}".format(sigma))
        _logits =[]

        for x_batch in x_test:
            idx_sigmas = tf.ones(x_batch.shape[0], dtype=tf.int32) * idx
            score = model([x_batch, idx_sigmas])
            _logits.append(score)

        _logits = tf.concat(_logits, axis=0)
        score_dict.append(tf.identity(_logits))

    return score_dict