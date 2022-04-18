from operator import index
import numpy as np
from unittest.mock import patch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from race import Race
from lsh_functions import SRPHash
import tensorflow as tf
import matplotlib.pyplot as plt


def get_image_generator(path, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    
    generator = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse')
    return generator


def indexInRace(race, image_generator, patch_size):
    images = image_generator.next()[0]
    total = images.shape[0]
    i = 0
    for image in images:
        print(f"Progress: {i}/{total}", end='\r')
        patches = extract_patches(image, patch_size)
        # reshape to one dimensional array of patches
        patches = tf.reshape(patches, (-1, patch_size * patch_size * 3))
        patches -= 127.5
        race.score(patches)
        i += 1
    print(f"Progress: {i}/{total}", end='\r')
    return race

def extract_patches(image, patch_size):
    return tf.image.extract_patches(images=tf.expand_dims(image, 0),
                           sizes=[1, patch_size, patch_size, 1],
                           strides=[1, patch_size, patch_size, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')

def makeRace(repetitions, concatenations, num_bits, buckets, patch_size, seed):
    hash_module = SRPHash(dimension=patch_size * patch_size * 3, num_hashes=repetitions * concatenations, num_bits=num_bits, seed=seed)
    return Race(repetitions, concatenations, buckets, hash_module)

def getImageScores(image, race, patch_size, plot=False):
    patches = extract_patches(image, patch_size)
    # reshape to one dimensional array of patches
    patches = tf.reshape(patches, (-1, patch_size * patch_size * 3))
    patches -= 127.5
    scores = race.get_score(patches)

    if plot:
        sorted_scores = sorted(scores)
        plt.plot(list(range(len(sorted_scores))), sorted_scores, 'o', color='black')

    return scores

def transformImage(image, patch_size, race_scores, threshold, show_transformed_image=False):
    with tf.GradientTape(persistent=True) as tape:
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tape.watch(image_tensor)
        patches = extract_patches(image_tensor, patch_size)
        patches = tf.Variable(tf.reshape(patches, (-1, patch_size * patch_size * 3)))
        for i in range(patches.shape.dims[0]):
            race_score = race_scores[i]
            if race_score > threshold:
                patches[i].assign(np.zeros(patches.shape.dims[1]))

        transformed_image = extract_patches_inverse(image_tensor, patches, patch_size, tape)

        if show_transformed_image:
            plt.axis("off")
            plt.imshow(transformed_image)
            plt.show()

        return transformed_image

def extract_patches_inverse(image, patches, patch_size, tape):
    _x = tf.zeros_like(image)
    _y = extract_patches(_x, patch_size)
    grad = tape.gradient(_y, _x)
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tape.gradient(_y, _x, output_gradients=patches) / grad
