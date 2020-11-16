import cv2 
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("imgs/minecraft.jpg")
for i in range(0, 3, 1):
  plt.hist(img[i].ravel(), 256, [0, 256])
img = cv2.imread("imgs/minecraft.jpg", 0)
hist = plt.hist(img.ravel(), 256, [0, 256])
plt.show()  