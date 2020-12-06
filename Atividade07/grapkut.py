from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import cv2
import numpy as np

class GrabCutGUI(Frame):
    # Instancia um frame novo
    def __init__(self, master = None):
        Frame.__init__(self, master)

        self.iniciaUI()
    # inicializa o frame
    def iniciaUI(self):
        
        self.master.title("Janela da Imagem Segmentada")
        self.pack()

        # Configura as callbacks       
        self.computaAcoesDoMouse()
        self.imagem = self.carregaImagemASerExibida()

        self.canvas = Canvas(self.master, width = self.imagem.width(), height = self.imagem.height(), cursor = "cross")

        self.canvas.create_image(0, 0, anchor = NW, image = self.imagem)
        self.canvas.image = self.imagem

        self.canvas.pack()

    # Callback para quando ocorre uma ação com o mouse
    def computaAcoesDoMouse(self):
        self.startX = None
        self.startY = None
        self.rect   = None
        self.rectangleReady = None
        
        self.master.bind("<ButtonPress-1>", self.callbackBotaoPressionado)
        self.master.bind("<B1-Motion>", self.callbackBotaoPressionadoEmMovimento)
        self.master.bind("<ButtonRelease-1>", self.callbackBotaoSolto)

    # Callback para quando o botão do mouse é solto.
    # Quando isso ocorre, significa que o retângulo foi traçado,
    # e agora já podemos aplicar o método.
    # Esta função aplica o grapkut e mostra na tela o resultado
    def callbackBotaoSolto(self, event):
        if self.rectangleReady:
            
            windowGrabcut = Toplevel(self.master)
            windowGrabcut.wm_title("Segmentation")
            windowGrabcut.minsize(width = self.imagem.width(), height = self.imagem.height())

           
            canvasGrabcut = Canvas(windowGrabcut, width = self.imagem.width(), height = self.imagem.height())
            canvasGrabcut.pack()

            
            mask = np.zeros(self.imagemOpenCV.shape[:2], np.uint8)
            print(mask.shape)
            rectGcut = (int(self.startX), int(self.startY), int(event.x - self.startX), int(event.y - self.startY))
            fundoModel = np.zeros((1, 65), np.float64)
            objModel = np.zeros((1, 65), np.float64)

            
            cv2.grabCut(self.imagemOpenCV, mask, rectGcut, fundoModel, objModel, 5, cv2.GC_INIT_WITH_RECT)

           
            maskFinal = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            maskFund = np.where((mask == 1) | (mask == 3), 0, 1).astype('uint8')
            """
                Aqui ele cria duas images, uma com o objeto, outra com o fundo.
                Então, aplica o blur na imagem do fundo, e finalmente, coloca as
                imagens juntas novamente.
                Para tanto, o algoritmo percorre pexel-a-pixel, e então verifica onde
                na máscara está zerado, pois 0 na máscara significa que o píxel pertence
                ao fundo, e cola o respectivo píxel do fundo na imagem original. No final,
                os pixels do objeto estarão intactos, mas os pixels que estavam vazios, agora
                possuem o fundo, mas com um efeito de blur aplicado
            """
            imgFinal = self.imagemOpenCV * maskFinal[:,:,np.newaxis]
            funFinal = self.imagemOpenCV * maskFund[:,:,np.newaxis]

            funFinal = cv2.blur(funFinal,(5,5))
            for x in range(0, self.imagemOpenCV.shape[1]):
                for y in range(0, self.imagemOpenCV.shape[0]):
                    if(maskFinal[y][x] == 0):
                        imgFinal[y][x] = funFinal[y][x]


            imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)
            imgFinal = Image.fromarray(imgFinal)
            imgFinal = ImageTk.PhotoImage(imgFinal)

            canvasGrabcut.create_image(0, 0, anchor = NW, image = imgFinal)
            canvasGrabcut.image = imgFinal    
  

    def callbackBotaoPressionadoEmMovimento(self, event):
        currentX = self.canvas.canvasx(event.x)
        currentY = self.canvas.canvasy(event.y)

        self.canvas.coords(self.rect, self.startX, self.startY, currentX, currentY)

        self.rectangleReady = True

    def callbackBotaoPressionado(self, event):
        self.startX = self.canvas.canvasx(event.x)
        self.startY = self.canvas.canvasy(event.y)

        if not self.rect:
            self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="blue")

    def carregaImagemASerExibida(self):
        caminhoDaImagem = tkinter.filedialog.askopenfile()
        if(not caminhoDaImagem is None):
            print(caminhoDaImagem)
            self.imagemOpenCV = cv2.imread(caminhoDaImagem.name)

            image = cv2.cvtColor(self.imagemOpenCV, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)

            image = ImageTk.PhotoImage(image)

            return image
            

def main():
    # Instancia o tkinter para criar o novo frame
    root = Tk()

    # Cria o novo frame
    appcut = GrabCutGUI(master = root)

    # Loop que será responsável por exibir e atualizar itens na tela
    appcut.mainloop()

# Chama a main
if __name__ == "__main__":
    main()
