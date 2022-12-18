'''
                        Cut patches of eyes and mouth

This program is used to crop the eye and mouth area.
After that, the images are saved in two folders: eyes and mouth

Camera: OpenCv
Face detector:      dlib -> HOG-SVM and CNN
Eye localization:   dlib -> shape_predictor
Mouth localization: dlib -> shape_predictor

@author: mjflores
@date:   20/12/2022
@ref: 
    https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
    https://medium.datadriveninvestor.com/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c
@models:
    https://github.com/davisking/dlib-models

python3==3.10.6   
opencv-contrib-python==4.5.5.64
opencv-python==4.5.5.64
'''

import numpy as np
import dlib
import cv2

colorEyeRectangle   = (200,10,200)
colorMouthRectangle = (10,200,10)

def get_left_right_eyes(sh):
   left_eye = []
   right_eye = []
   
   left_eye.append([sh.part(36).x,sh.part(36).y])
   left_eye.append([sh.part(37).x,sh.part(37).y])
   left_eye.append([sh.part(38).x,sh.part(38).y])
   left_eye.append([sh.part(39).x,sh.part(39).y])
   left_eye.append([sh.part(40).x,sh.part(40).y])
   left_eye.append([sh.part(41).x,sh.part(41).y])
   
   right_eye.append([sh.part(42).x,sh.part(42).y])
   right_eye.append([sh.part(43).x,sh.part(43).y])
   right_eye.append([sh.part(44).x,sh.part(44).y])
   right_eye.append([sh.part(45).x,sh.part(45).y])
   right_eye.append([sh.part(46).x,sh.part(46).y])
   right_eye.append([sh.part(47).x,sh.part(47).y])

   return np.array(left_eye), np.array(right_eye)

def plot_eye_rectangles(frm,lEye,rEye,k1):
    dxy = 5
    dwh = 7
    lftLeft = lEye[0,0] - dxy*2
    rgtLeft = lEye[3,0] + dxy 
    
    topLeft = int(0.5*(lEye[1,1] + lEye[2,1])) - dwh
    btmLeft = int(0.5*(lEye[4,1] + lEye[5,1])) + dwh
    
    lftRight = rEye[0,0] - dxy
    rgtRight = rEye[3,0] + dxy*2
    topRight = int(0.5*(rEye[1,1] + rEye[2,1])) - dwh
    btmRight = int(0.5*(rEye[4,1] + rEye[5,1])) + dwh
    
    cropped_lEye = frm[topLeft:btmLeft, lftLeft:rgtLeft]    
    cropped_rEye = frm[topRight:btmRight, lftRight:rgtRight]
    nomEyeleft = "ojo%d"%(2*k1-1)
    nomEyeright = "ojo%d"%(2*k1)
    # Save 
    cv2.imwrite(dirSave+"eyes/"+nomEyeleft+".png", cropped_lEye)
    cv2.imwrite(dirSave+"eyes/"+nomEyeright+".png", cropped_rEye)
    #View
    cv2.rectangle(frm, (lftLeft, topLeft), (rgtLeft, btmLeft), colorEyeRectangle,1)
    cv2.rectangle(frm, (lftRight, topRight), (rgtRight, btmRight), colorEyeRectangle,1)
    
    cv2.putText(frm,"Left",(lftLeft, topLeft-5),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,0,0))
    cv2.putText(frm,"Right",(lftRight, topRight-5),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,0,0))

def get_external_internal_mouth(sh):
   external_mouth = []
   internal_mouth = []
   
   external_mouth.append([sh.part(48).x,sh.part(48).y])
   external_mouth.append([sh.part(49).x,sh.part(49).y])
   external_mouth.append([sh.part(50).x,sh.part(50).y])
   external_mouth.append([sh.part(51).x,sh.part(51).y])
   external_mouth.append([sh.part(52).x,sh.part(52).y])
   external_mouth.append([sh.part(53).x,sh.part(53).y])
   external_mouth.append([sh.part(54).x,sh.part(54).y])
   external_mouth.append([sh.part(55).x,sh.part(55).y])
   external_mouth.append([sh.part(56).x,sh.part(56).y])
   external_mouth.append([sh.part(57).x,sh.part(57).y])
   external_mouth.append([sh.part(58).x,sh.part(58).y])
   external_mouth.append([sh.part(59).x,sh.part(59).y])         
   
   internal_mouth.append([sh.part(60).x,sh.part(60).y])
   internal_mouth.append([sh.part(61).x,sh.part(61).y])
   internal_mouth.append([sh.part(62).x,sh.part(62).y])
   internal_mouth.append([sh.part(63).x,sh.part(63).y])
   internal_mouth.append([sh.part(64).x,sh.part(64).y])
   internal_mouth.append([sh.part(65).x,sh.part(65).y])
   internal_mouth.append([sh.part(66).x,sh.part(66).y])
   internal_mouth.append([sh.part(67).x,sh.part(67).y])      

   return np.array(external_mouth), np.array(internal_mouth)

def plot_external_mouth_rectangle(frm,extMouth, k1):
    dxy = 8
    dwh = 9
    
    lft = extMouth[0,0] - 2*dxy
    rgt = extMouth[6,0] + 2*dxy
    top = int(0.5*(extMouth[2,1] + extMouth[4,1])) - dwh
    btm = int(0.34*(extMouth[8,1] + extMouth[9,1]+extMouth[10,1])) + dwh
    
    nomMount = "boca%d"%k1
    # Save 
    cropped_mouth = frm[top:btm, lft:rgt]
    cv2.imwrite(dirSave+"mouth/"+nomMount+".png", cropped_mouth)
           
    cv2.rectangle(frm, (lft, top), (rgt, btm), colorMouthRectangle,1)
    cv2.putText(frm,"Mouth",(lft, btm+6),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,0,0))
    
#===========================================================

dirSP     = "dirModels"
dirVideos = "dirYourVideos"
dirSave   = "dir eyes and mouth folders"

# Face detector
#detectorFace  = dlib.get_frontal_face_detector()
detectorFace   = dlib.cnn_face_detection_model_v1(dirSP+"mmod_human_face_detector.dat")

# Face landmarks
#shapePredictor = dlib.shape_predictor(dirSP+"shape_predictor_68_face_landmarks.dat")
shapePredictor = dlib.shape_predictor(dirSP+"shape_predictor_68_face_landmarks_GTX.dat")

vs = cv2.VideoCapture(dirVideos+"conductor2.avi") #initialize video capture
titleFrame = "Eyes and mouth"
cv2.namedWindow(titleFrame, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(titleFrame,200,100)

k = 1
# loop over frames from the video stream
while True:
    ret, frame = vs.read()
    if ret == False:
       print("Error capture image")
       break
       
    # This program works on 640x480 pixels
    if frame.shape[0] != 480 and frame.shape[0] != 640:
       frame = cv2.resize(frame, (640,480))
       
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # detect faces in the grayscale frame
    faces = detectorFace(gray, 0)
    # loop over the face detections
    for rectFace in faces:                   
        rectFace = rectFace.rect # uncomment with CNN        
        
        shape = shapePredictor(gray, rectFace)
        
        leftEye, rightEye = get_left_right_eyes(shape)      
        plot_eye_rectangles(frame,leftEye,rightEye,k)
              
        externalMouth, internalMouth = get_external_internal_mouth(shape)
        plot_external_mouth_rectangle(frame,externalMouth,k)
        
        k = k + 1    
        
    # show the frame
    cv2.imshow(titleFrame, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()
