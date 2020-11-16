import cv2

img = cv2.imread("imgs/j.png")
img = cv2.resize(img, (640, 480))

def videoErosao(img):
    out = cv2.VideoWriter("Erosao.avi", cv2.VideoWriter_fourcc("X", "V", "I", "D"), 20.0, (640, 480))
    for i in range(30*10):
        _i = int (i / 5)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, ((_i) + 1, (_i)+1), ((_i), (_i)))
        newFrame = cv2.erode(img, element)
        out.write(newFrame)

def videoDilatacao(img):
    out = cv2.VideoWriter("Dilatacao.avi", cv2.VideoWriter_fourcc("X", "V", "I", "D"), 20.0, (640, 480))
    for i in range(85*10):
        _i = int (i / 2)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, ((_i) + 1, (_i)+1), ((_i), (_i)))
        newFrame = cv2.dilate(img, element)
        out.write(newFrame)
videoDilatacao(img)
videoErosao(img)