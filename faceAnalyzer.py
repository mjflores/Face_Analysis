#------- Librerias para GUI ------------
import tkinter as tk
import cv2
from PIL import Image, ImageTk

#-------- Librerias para IA ------------
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle

#------- Subprogramas -----------------
from Emociones import emocion
from Liveness import liveness

#------- Carga de los Modelos ----------
# Rutas de los modelos
directorio_modelos = "Modelos"

ruta_model_liv  = directorio_modelos + "/liveness_model.h5"
ruta_le         = directorio_modelos + "/le.pickle"

#print("Cargando modelo Liveness...")
model_liv = load_model(ruta_model_liv)  
le = pickle.loads(open(ruta_le,"rb").read())
print("Modelo Liveness cargado...")

rutah5emociones = directorio_modelos + "/ResNet50_model.h5"
modelo_emociones = load_model(rutah5emociones)

print("Modelo FER cargado...")


face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
print("Modelo de reconocimiento inicializado.")

#--------- Constantes de ejecución --------------
diccionario_emocion = {0: "Enojo", 1: "Disgusto", 2: "Miedo", 3: "Feliz", 4: "Neutral", 5: "Triste", 6: "Sorpresa"}
img_size = 224      # Tamaño de la imagen para el pre-procesamiento
size = 1

def onClossing():
    root.quit()
    cap.release()
    print("Camera Disconnected")
    root.destroy()

def callback():
    ret, frame = cap.read()
    if not ret:
        onClossing()
    
    frame = cv2.flip(frame, 1)

    # grayscale image for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
    # get face region coordinates
    faces = face_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])

    if faces: # Para reconocimiento de un rostro a la vez por cuadro
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        roi_gray    = gray[y:y + h, x:x + w]
        roi_gray1   = np.array(roi_gray)

        # Colocar cuadro de rostro
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,200),2)

        # Detectar Liveness
        liv_label = liveness.detectar_liveness(model_liv,le,frame,x,y,w,h)
      
        cv2.putText(frame, liv_label, (x+5, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
      
        # Predicir emoción
        altura, emociones = emocion.predecir_emocion(modelo_emociones,roi_gray1,x,y,w,h)
  
     
        # Imprimir expresiones faciales
        for i in range(7):
            cv2.putText(frame, diccionario_emocion[i]+'='+str(emociones[i]), (x+w, altura+(i*15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emocion.color_emocion(i), 1, cv2.LINE_AA)
          

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label.configure(image=img)
    label.image = img
    root.after(32, callback)



cap = cv2.VideoCapture(0)


root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", onClossing)

root.title("Video")

label = tk.Label(root)
label.pack()

root.after(32, callback)
root.mainloop()
