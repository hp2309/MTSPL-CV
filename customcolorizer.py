import numpy as np
import cv2
import itertools


class Colorizer:
    frame = None
    img = None
    def __init__(self) -> None:
        pass
    def colorize(self, depthf, near_threshold, far_threshold):
        self.frame = depthf
        self.img = np.zeros((self.frame.shape[0], self.frame.shape[1], 3), dtype= np.uint8)
        delta = far_threshold-near_threshold
        f = 255.0/delta
        
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                for k in range(self.img.shape[2]):
                    self.img[i][j][k] = (far_threshold-self.frame[i][j])*f

        return self.img

# c = Colorizer()

# a = np.arange(120000)
# a = a.reshape(300,400)
# i = c.colorize(a)
# print(a)
# print(i)
# i = np.asanyarray(i)
# cv2.imshow("img", i)

# while True:
#     key = cv2.waitKey(1)
#     if key == 27:
#         cv2.destroyAllWindows()
#         break