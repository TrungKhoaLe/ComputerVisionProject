import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        # if the processors are None, initialize them as an empty
        # list
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose = -1):
        # initialize the lists of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # assuming the format of imagePath is 
            # path/to/dataset/{label}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
            # show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format((i + 1),
                len(imagePaths))
        return (np.array(data), np.array(labels))