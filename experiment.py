from operator import index
from unittest.mock import patch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from race import Race
from lsh_functions import SRPHash
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_images(path, n_samples):
    datagen = ImageDataGenerator()
    
    generator = datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=n_samples,
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
        # NOTE: center
        patches -= 127.5
        # NOTE
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
    # NOTE: center
    patches -= 127.5
    # NOTE
    scores = race.score(patches)
    # scores = []
    # for patch in extract_patches(image, patch_size):
    #     scores.append(race.get_score(patch))

    if plot:
        sorted_scores = sorted(scores)
        plt.plot(list(range(len(sorted_scores))), sorted_scores, 'o', color='black')

    return scores

def transformImage(image, patch_size, race_scores, threshold, show_transformed_image=False):
    patches = extract_patches(image, patch_size)
    for i in range(len(patches)):
        race_score = race_scores[i]
        if race_score > threshold:
            patches[i] = [0 for _ in range(len(patches[0]))]

    transformed_image = extract_patches_inverse(image, patches)

    if show_transformed_image:
        plt.axis("off")
        plt.imshow(transformed_image)
        plt.show()

#     return transformed_image

def extract_patches_inverse(image, patches):
    _x = tf.zeros_like(image)
    _y = extract_patches(_x)
    grad = tf.gradients(_y, _x)[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tf.gradients(_y, _x, grad_ys=patches)[0] / grad
