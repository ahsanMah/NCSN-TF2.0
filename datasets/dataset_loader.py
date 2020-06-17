import os

import tensorflow as tf
import tensorflow_datasets as tfds

import configs
import utils

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data(dataset_name):
    # load data from tfds

    if dataset_name in ["masked_fashion", "blown_fashion", "blown_masked_fashion"]:
        dataset_name="fashion_mnist"

    data_generators = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", shuffle_files=False)
    train = tf.data.Dataset.from_tensor_slices(data_generators['train']['image'])
    test = tf.data.Dataset.from_tensor_slices(data_generators['test']['image'])
    return train, test

# @tf.function
# def concat_mask(x):
#     mask = tf.cast(x>0, dtype=tf.float32)
#     return tf.concat((x,mask), axis=-1)

def preprocess(dataset_name, data, train=True):
    data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)  # rescale [0,255] -> [0,1]
    
    if dataset_name in ["blown_fashion", "blown_masked_fashion"]:
        data = data.map(lambda x: tf.image.pad_to_bounding_box(
                x, offset_height=14, offset_width=14,
                target_height=56, target_width=56
            ))

    if dataset_name in ["masked_fashion", "blown_masked_fashion"]:
        data = data.map(lambda x: tf.concat(
                (x, tf.cast(x>0, dtype=tf.float32)),
                axis=-1)
            )
    
    if train and dataset_name in ["cifar10"]:
        data = data.map(lambda x: tf.image.random_flip_left_right(x),
                        num_parallel_calls=AUTOTUNE)  # randomly flip along the vertical axis

    return data


def _preprocess_celeb_a(data, random_flip=True):
    # Discard labels and landmarks
    data = data.map(lambda x: x['image'], num_parallel_calls=AUTOTUNE)
    # Take a 140x140 centre crop of the image
    data = data.map(lambda x: tf.image.resize_with_crop_or_pad(x, 140, 140), num_parallel_calls=AUTOTUNE)
    # Resize to 32x32
    data = data.map(lambda x: tf.image.resize(x, (32, 32)), num_parallel_calls=AUTOTUNE)
    # Rescale
    data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)
    # Maybe cache in memory
    # data = data.cache()
    # Randomly flip
    if random_flip:
        data = data.map(lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=AUTOTUNE)
    return data


def get_celeb_a(random_flip=True):
    batch_size = configs.config_values.batch_size
    data_generators = tfds.load(name='celeb_a', batch_size=batch_size, data_dir="data", shuffle_files=True)
    train = data_generators['train']
    test = data_generators['test']
    train = _preprocess_celeb_a(train, random_flip=random_flip)
    test = _preprocess_celeb_a(test, random_flip=False)
    return train, test


def get_celeb_a32():
    """
    Loads the preprocessed celeb_a dataset scaled down to 32x32
    :return: tf.data.Dataset with single batch as big as the whole dataset
    """
    path = './data/celeb_a32'
    if not os.path.exists(path):
        print(path, " does not exits")
        return None

    images = utils.get_tensor_images_from_path(path)
    data = tf.data.Dataset.from_tensor_slices(images)
    data = data.map(lambda x: tf.cast(x, tf.float32))
    data = data.batch(int(tf.data.experimental.cardinality(data)))
    return data

def get_ood_data(dataset_name):
    print("Getting OOD Dataset...")
    OOD_LABEL = 0
    data = tfds.load(name="mnist", batch_size=-1, data_dir="data", shuffle_files=False)
    
    mask = data["train"]["label"] != OOD_LABEL
    inlier_train = tf.data.Dataset.from_tensor_slices(data["train"]["image"][mask])
    
    mask = data["test"]["label"] != OOD_LABEL
    inlier_test = tf.data.Dataset.from_tensor_slices(data["test"]["image"][mask])

    mask = data["test"]["label"] == OOD_LABEL
    ood_test = tf.data.Dataset.from_tensor_slices(data["test"]["image"][mask])

    inlier_train = preprocess("mnist", inlier_train, train=True)
    inlier_test = preprocess("mnist", inlier_test, train=False)
    ood_test = preprocess("mnist", ood_test, train=False)

    return inlier_train, inlier_test, ood_test

def get_train_test_data(dataset_name):

    if dataset_name == 'mnist_ood':
        train,test,_ = get_ood_data(dataset_name)
    elif dataset_name != 'celeb_a':
        train, test = load_data(dataset_name)
        train = preprocess(dataset_name, train, train=True)
        test = preprocess(dataset_name, test, train=False)
    else:
        train, test = get_celeb_a()
    return train, test


def get_data_inpainting(dataset_name, n):
    if dataset_name == 'celeb_a':
        data = get_celeb_a(random_flip=False)[0]
        data = next(iter(data.take(1)))[:n]
    else:
        data_generator = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", split='train', shuffle_files=True)
        data = data_generator['image']
        data = tf.random.shuffle(data, seed=1000)
        data = data[:n] / 255
    return data


def get_data_k_nearest(dataset_name):
    data_generator = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", split='train', shuffle_files=False)
    data = tf.data.Dataset.from_tensor_slices(data_generator['image'])
    data = data.map(lambda x: tf.cast(x, dtype=tf.float32))

    return data
