'''
         Detector de rostros usando SSD


El objetivo es acelerar la captura de imagenes y su visualizacion, por lo tanto, no usa cv2.imshow()
En su lugar, para visualizar la imagen se usa TKinter


Ref: 
   https://www.tutorialspoint.com/using-opencv-with-tkinter
   https://stackoverflow.com/questions/32342935/using-opencv-with-tkinter

'''
#------- Librerias para GUI ------------
# Import required Libraries
from tkinter import Tk, Label
from PIL import Image, ImageTk
import cv2

import numpy as np
import time


dirSSD   ="path SSD"
dirVideo ="path Video"



detectorSSD = cv2.dnn.readNetFromCaffe(dirSSD+"deploy.prototxt" ,dirSSD+"res10_300x300_ssd_iter_140000.caffemodel")


FPS = 30

resizeW, resizeH = 320, 240

def SSD_2_rectangles(detections, th):
    faces = []
    for i in range(detections[0][0].shape[0]):
      if(detections[0][0][i][2]>th):
        #print("Parametros ", detections[0][0][i])
        x1 = int(resizeW*detections[0][0][i][3])
        y1 = int(resizeH*detections[0][0][i][4])
        r1 = int(resizeW*detections[0][0][i][5])
        b1 = int(resizeH*detections[0][0][i][6])
        faces.append([x1,y1,r1-x1,b1-y1])
    return faces

#==============================================================

# Create an instance of TKinter Window or frame
win = Tk()
win.wm_title("Video capture TK")
win.config(background="#FFFFFF")

# Set the size of the window
win.geometry("640x480")# Create a Label to capture the Video frames

label = Label(win)
label.grid(row=0, column=0)

# Define function to show frame
def show_frames():
      tic = time.time()
      ret, frame = cap.read()
      if not ret:
        sys.exit()
      #print("frame",type(frame))
      # Get the latest frame and convert into Image
      imgRGB  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

      baseImage = cv2.resize( imgRGB, (resizeW, resizeH))
      imageBlob = cv2.dnn.blobFromImage(image = baseImage)

      detectorSSD.setInput(imageBlob)
      faces = SSD_2_rectangles(detectorSSD.forward(),0.70)
      for (_x,_y,_w,_h) in faces:
        x = 2*_x
        y = 2*_y
        w = 2*_w
        h = 2*_h
        cv2.rectangle(imgRGB, (x,y),(x+w,y+h),(50,50,200),2 )

      # Convert array to image
      img = Image.fromarray(imgRGB)
      # Convert image to PhotoImage
      imgtk = ImageTk.PhotoImage(image = img)
      label.imgtk = imgtk
      label.configure(image=imgtk)
      toc = time.time()
      print("Elapse time %0.3f " % (1000*(toc-tic)),"ms")
      #Repeat after an interval to capture continiously
      label.after(FPS, show_frames)

#==============================================================

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(dirVideo + "myVideo.avi")


show_frames()
win.mainloop()
