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
    for image in image_generator.next():
        patches = extract_patches(image, patch_size)
        race.score(patches)
        
    return race

def extract_patches(image, patch_size):
    return tf.image.extract_patches(images=[image],
                           sizes=[1, patch_size, patch_size, 3],
                           strides=[1, patch_size, patch_size, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')


repetitions = 100
concatenations = 2
buckets = 1_000_000
hash_module = SRPHash(dimension=20)
race = Race( )

def getImageScores(image, race, patch_size, plot=False):
    scores = []
    for patch in extract_patches(image, patch_size):
        scores.append(race.getScore(patch))

    if plot:
        sorted_scores = sorted(scores)
        plt.plot(list(range(len(sorted_scores))), sorted_scores, 'o', color='black');

    return scores

def transformImage(image, patch_size, race_scores, threshold, show_transformed_image=False):
    transformed_image = image
    for i, patch in enumerate(extract_patches(image, patch_size)):
        race_score = race_scores[i]
        if race_score > threshold:
            zero out that part of the image
    
    if show_transformed_image:
        plt.axis("off")
        plt.imshow(transformed_image)
        plt.show()

    return transformed_image



# def main():
#     pass