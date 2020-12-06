import cv2
import numpy as np

import numpy as np
import cv2

def computeKmeans(image, k=256):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    
    _, labels, centroides = cv2.kmeans(Z, k, None, criteria, 40, cv2.KMEANS_RANDOM_CENTERS)
    
    centroides = np.uint8(centroides)
    imagemColoridaComCentroides = centroides[labels.flatten()]
    imagemFinal = imagemColoridaComCentroides.reshape((image.shape))
    
    cv2.imshow("original", image)
    cv2.imshow("resultado kmeans", imagemFinal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = cv2.imread('imgs/tux.jpg')
    computeKmeans(image, 8)
    computeKmeans(image, 256)

