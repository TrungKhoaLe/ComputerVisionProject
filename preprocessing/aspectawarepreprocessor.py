import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self, image):
        # pick up the dimensions of the image
        # dW, dH are used to crop the image
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        if w < h:
            image = imutils.resize(image, width=self.width,
            inter=self.inter)
            dH = int((image.shape[0] - self.height)/2.)
        else:
            image = imutils.resize(image, height=self.height,
            inter=self.inter)
            dW = int((image.shape[1] - self.width)/2.)
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        return cv2.resize(image, (self.width, self.height),
        interpolation=self.inter)
