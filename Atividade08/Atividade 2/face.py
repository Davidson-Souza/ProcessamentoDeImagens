from tqdm import tqdm
import face_recognition
import pickle
import cv2

video_filename = "vid.mp4"
#carrega arquivo binário contendo faces codificadas
data_encoding = pickle.loads(open("face_encodings", "rb").read())

#carrega vídeo do disco
videoCaptureInput = cv2.VideoCapture(video_filename)
#set contendo as possíveis pessoas reconhecidas
unique_names = set(data_encoding["names"])

#gerador de vídeo contendo saída com faces reconhecidas
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
fps = videoCaptureInput.get(cv2.CAP_PROP_FPS)

videoCaptureOutput = cv2.VideoWriter("output.mp4", fourcc, fps, (1920, 1080))

#gera reconhecimento em vídeo para os 200 primeiros frames
for i in tqdm(range(0, 2)):
    #para cada frame
    success, frame = videoCaptureInput.read()
        
    #acabou o vídeo?
    if success == False:
        break
        
    #converte frame de formato BGR (OpenCV) para RGB (face_recognition)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(frame, model = 'cnn')
    encodings = face_recognition.face_encodings(frame, boxes)
    
    names = []

    #para cada codificação de faces encontrada
    for encoding in encodings:
        matches = face_recognition.compare_faces(data_encoding["encodings"], encoding)
        
        #retorna o identificador da lista das faces da base que "batem" com a codificação verificada
        matchesId = [i for i, value in enumerate(matches) if value == True]
        
        #faz uma espécie de "votação": quem tiver mais codificações "próximas" das faces treinadas na base "ganha"
        counts = {}
        for name in unique_names:
            counts[name] = 0  
        for i in matchesId:
            name = data_encoding["names"][i]
            counts[name] += 1
        name = max(counts, key = counts.get)
        names.append(name)

    #desenha o retângulo e escreve o nome da pessoa no frame
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
        cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
    
    #converte o frame de volta pro formato do OpenCV (BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #escreve o frame no arquivo de vídeo
    videoCaptureOutput.write(frame)
print("Acabou!")

videoCaptureInput.release()
videoCaptureOutput.release()