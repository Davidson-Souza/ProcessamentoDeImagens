import cv2
from skimage import feature
from sklearn.svm import SVC
import numpy as np
import pickle
from tqdm import tqdm
from os import listdir
import sys
import cv2
import numpy as np

from os import listdir
from os.path import isfile, join
objs = ["saudável", "bicho minerio", "ácaro branco", "ácaro vermelho", "ácaro de mancha anelar", "acaro_vermelho",\
        "broca", "cercosporiose", "cigarra", "cochonilhas farinhentas", "cochonilhas pardas", \
        "ferrugem" ]
usage = '===============================================\n \
Para classificar uma imagem de uma folha de café, basta passar o nome da imagem após o nome\
do executável. \n python {} [imagem].[extenção]\
Note que o modelo deve estar na mesma pasta, ou então deve ser passado como segundo argumento \n \
\n python {} [nome_do_modelo].svm'

svmFilename = "model.svm"

widthWindow = 400
heightWindow = 400

orientationsParam = 9
pixelsPerCellParam = 4
cellsPerBlockParam = 2

def testImages(svmObj, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("Classificando a imagem.")
    h, w, _ = img.shape
    img = cv2.resize(img, (widthWindow, heightWindow))

    H = feature.hog(img, orientations=orientationsParam, pixels_per_cell=(pixelsPerCellParam, pixelsPerCellParam), cells_per_block=(cellsPerBlockParam, cellsPerBlockParam), block_norm='L2-Hys', feature_vector = True)
    X = np.array(H)
    X = np.reshape(X, (-1, X.shape[0]))
    predict = svmObj.predict(X)
    print("A planta parece estar {}".format( objs[predict[0]] if predict == 0 else "com " +  objs[predict[0]]))

    
def extractFilenamesFromFolder(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles
def loadSVM():
    return pickle.load(open(svmFilename, "rb"))

def main():
    if (len (sys.argv) < 2):
        print(usage.format(sys.argv[0]))
        exit(1)
    img = cv2.imread(sys.argv[1])
    
    if img is None:
        print("Erro ao ler a imagem")
        print(usage.format(sys.argv[0]))
        exit(2)
    if (len(sys.argv) == 3):
        svmFilename = sys.argv[2]
        print("Usando o modelo {}".format(sys.argv[2]))
    
    svmObj = loadSVM()
    if svmObj is None:
        print("Erro ao ler o modelo")
        print(usage.format(sys.argv[0]))
        exit(3)
    print("===================================================")
    testImages(svmObj, img)
    print("===================================================")
    
if __name__ == "__main__":
    main()