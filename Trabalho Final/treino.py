import re
import cv2
from skimage import feature
from sklearn.svm import SVC
import numpy as np
import pickle
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

rootFolder = "dataset/"
trainFolder = "Train/"
testFolder = "Test/"
objs = ["saudavel", "bicho_mineiro", "acaro_branco", "acaro_branco", "acaro_mancha_redonda", "acaro_vermelho",\
        "broca", "cercosporiose", "cigarra", "cochonilhas_farinhentas", "cochonilhas_pardas", \
        "ferrugem_caffeiro" ]

cropTrainFolder = "Finished/"
cropTestFolder = "test/"

svmFilename = "model.svm"

widthWindow = 400
heightWindow = 400

orientationsParam = 9
pixelsPerCellParam = 4
cellsPerBlockParam = 2

def extractFilenamesFromFolder(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

def cropRegionOfEachPosImage(path, trainFolder, cropFolder, listOfImages, isTrain):
    i = 0
    for eachImage in tqdm(listOfImages):
        img = cv2.imread(path + trainFolder + objs[isTrain] + "/" + eachImage)

        img = cv2.resize(img, (widthWindow, heightWindow))
        flipImg = cv2.flip(img, 1)
        
        stringImg = listOfImages[i].split("/", 1)
        
        try:
            cv2.imwrite(path + cropFolder + objs[isTrain] + "/" + stringImg[0], img)
            cv2.imwrite(path + cropFolder +  objs[isTrain] + "/" +  "flip_" + str(i) + ".png", flipImg)
        except:
            print("Erro!")
        i += 1

def setDatabase():
    count = 0
    for i in tqdm(objs):
        vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + trainFolder + i + "/")
        cropRegionOfEachPosImage(rootFolder, trainFolder, cropTrainFolder, vectorOfFilenameImagesPos, count)
        count += 1
   

def trainSVM():
    X = np.empty(0)
    Y = np.empty(0)
    count = 0
    dimensionOfFeatureVector = 0
    print("Extraíndo features")
    for i in  objs:
        vectorOfFilenameImagesPos = extractFilenamesFromFolder(rootFolder + cropTrainFolder + i + "/")
        for eachImage in tqdm(vectorOfFilenameImagesPos):
            eachImage = eachImage.split("/", -1)
            eachImage = rootFolder + cropTrainFolder + i + "/" + eachImage[0]
        
            img = cv2.imread(eachImage, 0)
            img = cv2.resize(img, (widthWindow, heightWindow))
            H = feature.hog(img, orientations=orientationsParam, pixels_per_cell=(pixelsPerCellParam, pixelsPerCellParam), cells_per_block=(cellsPerBlockParam, cellsPerBlockParam), block_norm='L2-Hys', feature_vector = True)
            H = np.array(H)
        
            if(X.shape[0] == 0):
                X = H
                dimensionOfFeatureVector = X.shape[0]
            else:
                X = np.append(X, H)
            
            if(Y.shape[0] == 0):
                Y = np.array([0])
            else:
                Y = np.append(Y, np.array([count]))
        count += 1
    print(X.shape[0])
    print("Treinando o SVM.\n")
    X = np.reshape(X, (-1, dimensionOfFeatureVector))

    svm = SVC(kernel='linear')
    svm.fit(X, Y)
    return svm

def saveSVM(obj):
    pickle.dump(obj, open(svmFilename, "wb"))


def main():
    
    #constroi a base de dados
    print("Preparando o dataset")
    #setDatabase()
    
    #treina o SVM
    print("Treinando o modelo")
    svmObj = trainSVM()
    saveSVM(svmObj)
    
if __name__ == "__main__":
    main()