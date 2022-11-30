from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2 



def detectar_liveness(model_liv,le,img,x,y,w,h): 
    # model_liv -- modelo de reconocimiento de vida (*.h5)
    # le -- Codificador de etiquetas de clase (*.pickle)
    # img -- cuadro de la imagen obtenida del video
    # x,y,w,h -- coordenadas del cuadro del rostro, y ancho y alto
	
    # ===============================         
    # Extrae la ROI del rostro y luego la prepocesa exactamente
    # de la misma manera que los datos de entrenamiento
	face_2 = img[y:y+h, x:x+w]
	face_2 = cv2.resize(face_2, (48, 48)) 
	face_2 = face_2.astype("float")/255.0
	face_2 = img_to_array(face_2)
	face_2 = np.expand_dims(face_2,axis=0)

    # Pasar el ROI de la cara a través del modelo detector de vida
    # entrenado para determinar si la cara es "real" o "falsa"
	preds = model_liv.predict(face_2)[0] 
	j = np.argmax(preds)
	label = le[j]
    
    # Dibuja la etiqueta y el cuadro delimitador en el marco
	label = "{}: {:.4f}".format(label, preds[j])   

	return label # Retorna la etiqueta de "Real" o #Fake#
