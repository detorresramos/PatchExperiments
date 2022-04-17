from operator import index
from unittest.mock import patch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from race import Race
from lsh_functions import SRPHash
import tensorflow as tf


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
        patches = tf.image.extract_patches(images=[image],
                           sizes=[1, patch_size, patch_size, 3],
                           strides=[1, patch_size, patch_size, 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
        race.score(patches)
        
    return race


repetitions = 100
concatenations = 2
buckets = 1_000_000
hash_module = SRPHash(dimension=20)
race = Race( )

# def getImageScores(image, index, patch_size):
#     scores = []
#     for each patch of size patch_size in image:
#         query index with that patch and get score
#         scores.append(score)
#     do some sorting, graphing, etc
#     return scores (not sorted)

# def transformImage(race_scores, threshold, patch_size):
#     for each patch of size patch_size:
#         get race_score of that patch
#         if race_score higher than threshold:
#             that patch is not in image
#         else:
#             it is
#     show the image with the patches removed
#     return transformed image



# def main():
#     pass