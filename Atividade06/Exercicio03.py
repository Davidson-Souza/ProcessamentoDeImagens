import cv2 as cv
import numpy as np
import math

# Constantes globais para o CV2
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
channels = [0, 1]

# Calcula a distância entre as imagens
def calcDist(img1, img2):
    corr = cv.compareHist(img1, img2, cv.HISTCMP_CORREL)
    chiSquare = cv.compareHist(img1, img2, cv.HISTCMP_CHISQR)
    bhat = cv.compareHist(img1, img2, cv.HISTCMP_BHATTACHARYYA)
    return math.sqrt(math.pow(corr, 2) + math.pow(chiSquare, 2)  + math.pow(bhat, 2))

# Retorna o histograma de cada imagem
def calcHist(img):
    global channels, histSize, ranges
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    return hist

# Compara uma imagem com um conjunto de imagens, retorna o índice da mais parecida
def compara(base, imgSet):
    _dist = []
    index = 0
    base = calcHist(base)
    for i in imgSet:
        _dist.append([calcDist(base, calcHist(i)),index])
        index += 1
    _dist.sort()
    return _dist[0][1]

imgSet = []

# O nome das imgens vem aqui, deve ser idêntico ao nome do arquivo, com a mesma extenção
# A imagem base fica por último!
imgList = ["doofenshmirtz.jpg", "abacaxi2.jpg", "perry.jpg","minecraft.jpg", "abacaxi1.jpg"]
for i in imgList:
    imgSet.append (cv.imread("imgs/"+i))
    if imgSet[-1] is None:
        print("Erro ao ler a imagem", i)
        exit(0)

# Procura pela imagem que mais se parece com a última do vetor
base = imgSet.pop()
img = compara(base, imgSet)

# Mostra as duas imagens
cv.imshow("Original", base)
cv.imshow("imagem", imgSet[img])
cv.waitKey()