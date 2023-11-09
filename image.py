import cv2
import numpy as np
from matplotlib import pyplot as plt


class Image:

    def __init__(self, path=None, image=None):
        try:
            if image is None:
                self.image = cv2.imread(path)

                if self.image is None or self.image.size == 0:
                    raise ValueError()
                
            elif path is None:
                self.image = image

                if self.image is None or self.image.size == 0:
                    raise ValueError()

                self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            else:
                raise ValueError()
        
        except Exception as e:
            print(str(e))
        

    def __str__(self):
        return self.image

    def setImage(self, img):
        self.image = img

    def setGray(self, grayscale):
        self.gray = grayscale

    def sobelX(self, ksize=5):
        return cv2.Sobel(self.gray, cv2.CV_8U, 1, 0, ksize)

    def sobelY(self, ksize=3):
        return cv2.Sobel(self.gray, cv2.CV_8U, 0, 1, ksize)

    def probHoughTransform(self, edges, minLineLength, MaxLineGap, threshold=20):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, MaxLineGap)
        return lines