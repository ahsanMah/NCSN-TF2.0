import tensorflow as tf
from model.refinenet import RefineNet
import utils
import configs
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
from PIL import Image
import numpy as np


def clamped(x):
    return tf.clip_by_value(x, 0, 1.0)


def plot_grayscale(image):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.show()


def save_image(image, dir):
    plt.imshow(image, cmap=plt.get_cmap("gray"))
    plt.savefig(dir)


def save_as_grid(images, filename, spacing=2):
    """
    Partially from https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    :param images:
    :return:
    """
    # Define grid dimensions
    n_images, height, width, channels = images.shape
    rows = np.floor(np.sqrt(n_images)).astype(int)
    cols = n_images // rows

    # Init image
    grid_cols = rows * height + (rows + 1) * spacing
    grid_rows = cols * width + (cols + 1) * spacing
    im = Image.new('L', (grid_rows, grid_cols))
    for i in range(n_images):
        row = i // rows
        col = i % rows
        row_start = row * height + (1 + row) * spacing
        col_start = col * width + (1 + col) * spacing
        im.paste(tf.keras.preprocessing.image.array_to_img(images[i]), (row_start, col_start))
        # im.show()

    im.save(filename, format="PNG")


@tf.function
def sample_one_step(model, x, idx_sigmas, image_size, alpha_i):
    z_t = tf.random.normal(shape=image_size, mean=0, stddev=1.0)  # TODO: check if stddev is correct
    score = model([x, idx_sigmas])
    noise = tf.sqrt(alpha_i * 2) * z_t
    return x + alpha_i * score + noise


def sample(model, sigmas, eps=2 * 1e-5, T=100, n_images=1):
    """
    Only for MNIST, for now.
    :param model:
    :param sigmas:
    :param eps:
    :param T:
    :return:
    """
    image_size = (n_images, 28, 28, 1)

    x = tf.random.uniform(shape=image_size)
    # plot_grayscale(x[0, :, :, 0])

    for i, sigma_i in enumerate(sigmas):
        print(f"sigma {i + 1}/{len(sigmas)}")
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        idx_sigmas = tf.ones(n_images, dtype=tf.int32) * i
        for t in tqdm(range(T)):
            x = sample_one_step(model, x, idx_sigmas, image_size, alpha_i)

            if (t + 1) % 10 == 0:
                save_as_grid(x, samples_directory + f'sigma{i + 1}_t{t + 1}.png')
                # for j, sample in enumerate(x):
                #     img = Image.fromarray((plt.get_cmap("gray")(sample[:, :, 0]) * 255).astype(np.uint8))
                #     img.save(samples_directory + f'sample_{j}_{i + 1}.png')
                # save_image(sample[:, :, 0], samples_directory + f'sample_{j}_{i+1}.png')
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    latest_checkpoint = tf.train.latest_checkpoint(model_directory + dataset)
    print("loading model from checkpoint ", latest_checkpoint)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt.restore(latest_checkpoint)

    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(1.0),
                                           tf.math.log(0.01),
                                           10))

    samples = sample(model, sigma_levels, T=100, n_images=400)

    # for i, sample in enumerate(samples):
    #     # plot_grayscale(sample[:, :, 0])
    #     save_image(sample[:, :, 0], samples_directory + f'sample_{i}.png')
