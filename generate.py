import tensorflow as tf
from model.refinenet import RefineNet
import utils
import configs
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os

def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


def plot_grayscale(image):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()


def save_image(image, dir):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.savefig(dir)

def sample(model, sigmas, eps=2 * 1e-5, T=100, n_images=1):
    """
    Only for MNIST, for now.
    :param model:
    :param sigmas:
    :param eps:
    :param T:
    :return:
    """
    with tf.device('/GPU:0'):
        image_size = (n_images, 28, 28, 1)

        x = tf.random.uniform(shape=image_size)
        # plot_grayscale(x[0, :, :, 0])

        for i, sigma_i in enumerate(sigmas):
            print(f"sigma {i}/{len(sigmas)}")
            alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
            idx_sigmas = tf.ones(n_images, dtype=tf.int32) * i
            for t in tqdm(range(T)):
                z_t = tf.random.normal(shape=image_size, mean=0, stddev=1.0)  # TODO: check if stddev is correct
                score = model([x, idx_sigmas])
                noise = tf.sqrt(alpha_i * 2) * z_t
                x = x + alpha_i * score + noise

            # plot_grayscale(clamped(x[0, :, :, 0]))

            for j, sample in enumerate(x):
                save_image(sample[:, :, 0], samples_directory + f'sample_{j}_{i+1}.png')
        return x


if __name__ == '__main__':
    args = utils.get_command_line_args()
    configs.config_values = args

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    model_directory = './saved_models/'
    dataset = 'mnist/'
    samples_directory = './samples/' + start_time + "/"
    os.makedirs(samples_directory)

    step = tf.Variable(0)
    model = RefineNet(filters=16, activation=tf.nn.elu)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    latest_checkpoint = tf.train.latest_checkpoint(model_directory + dataset)
    print("loading model from checkpoint ", latest_checkpoint)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt.restore(latest_checkpoint)

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(1.0),
                                           tf.math.log(0.01),
                                           10))

    samples = sample(model, sigma_levels, T=100, n_images=30)

    # for i, sample in enumerate(samples):
    #     # plot_grayscale(sample[:, :, 0])
    #     save_image(sample[:, :, 0], samples_directory + f'sample_{i}.png')
