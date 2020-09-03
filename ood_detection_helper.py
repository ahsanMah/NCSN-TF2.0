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

    # print("=" * 20 + "\nParameters: \n")
    # for key in parser.__dict__:
    #     print(key + ': ' + str(parser.__dict__[key]))
    # print("=" * 20 + "\n")
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


def ood_metrics(inlier_score, outlier_score, plot=False, verbose=False, 
                names=["Inlier", "Outlier"]):
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import roc_curve
    from sklearn.metrics import classification_report, average_precision_score
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    

    y_true = np.concatenate((np.zeros(len(inlier_score)),
                             np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))
    
    prec_in, rec_in, _ = precision_recall_curve(y_true, y_scores)

    # Outliers are treated as "positive" class 
    # i.e label 1 is now label 0
    prec_out, rec_out, _ = precision_recall_curve((y_true==0), -y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
    tpr95_idx = np.where(np.isclose(tpr,0.95))[0][0]
    tpr80_idx = np.where(np.isclose(tpr,0.8))[0][0]

    metrics = dict(
        roc_auc = roc_auc_score(y_true,y_scores),
        fpr_tpr95 = fpr[tpr95_idx],
        fpr_tpr80 = fpr[tpr80_idx],
        pr_auc_in = auc(rec_in, prec_in),
        pr_auc_out = auc(rec_out, prec_out),
        ap = average_precision_score(y_true,y_scores)
    )
    
    if plot:    
    
        fig, axs = plt.subplots(1,2, figsize=(16,4))
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=True)
        sns.lineplot(fpr, tpr, ax=axs[0])
        axs[0].set(
            xlabel="FPR", ylabel="TPR", title="ROC", ylim=(-0.05, 1.05)
        )

        sns.lineplot(rec_in, prec_in, ax=axs[1], label="PR-In")
        sns.lineplot(rec_out, prec_out, ax=axs[1], label="PR-Out")
        axs[1].set(
            xlabel="Recall", ylabel="Precision", title="Precision-Recall", ylim=(-0.05, 1.05)
        )
        fig.suptitle("{} vs {}".format(*names), fontsize=20)
        plt.show()
        plt.close()
    
    if verbose:
        print("{} vs {}".format(*names))
        print("----------------")
        print("ROC-AUC: {:.4f}".format(metrics["roc_auc"]))
        print("PR-AUC (In/Out): {:.4f} / {:.4f}".format(metrics["pr_auc_in"], metrics["pr_auc_out"]))
        print("FPR (95% TPR) Prec: {:.4f}".format(metrics["fpr_tpr95"]))
        
    return metrics


def evaluate_model(train_score, test_score, outlier_score, outlier_score_2, labels):
    fig, axs = plt.subplots(2,1, figsize=(16,8))
    colors = ["red", "blue", "green", "orange"]

    sns.distplot(train_score,color=colors[0], label="Training", ax=axs[0])
    sns.distplot(test_score, color=colors[1], label=labels[1], ax=axs[0])
    sns.distplot(outlier_score, color=colors[2], label=labels[2], ax=axs[0])
    sns.distplot(outlier_score_2, color=colors[3], label=labels[3], ax=axs[0])

    sns.distplot(test_score, color=colors[1], label=labels[1], ax=axs[1])
    sns.distplot(outlier_score, color=colors[2], label=labels[2], ax=axs[1])

    axs[0].legend()
    axs[1].legend()
    plt.show()
    
    ood_metrics(-test_score, -outlier_score_2, names=(labels[1], labels[3]),
                plot=False, verbose=True)
    print()
    ood_metrics(-test_score, -outlier_score, names=(labels[1], labels[2]),
                plot=False, verbose=True)
    return