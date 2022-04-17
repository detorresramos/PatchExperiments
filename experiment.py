

from operator import index
from unittest.mock import patch


def createRACE(patch_size):
    init race index
    for each image in images:
        for each patch of size patch_size in image:
            add to race index
    return index

def getImageScores(image, index, patch_size):
    scores = []
    for each patch of size patch_size in image:
        query index with that patch and get score
        scores.append(score)
    do some sorting, graphing, etc
    return scores (not sorted)

def transformImage(race_scores, threshold, patch_size):
    for each patch of size patch_size:
        get race_score of that patch
        if race_score higher than threshold:
            that patch is not in image
        else:
            it is
    show the image with the patches removed
    return transformed image



def main():
    pass