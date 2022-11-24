# Face detector based on SSD
import cv2

dirSDD = "poner la direccion del archivo"
dirVideo = "poner la direccion del video"

detectorSSD = cv2.dnn.readNetFromCaffe(dirSDD+"deploy.prototxt" ,dirSDD+"res10_300x300_ssd_iter_140000.caffemodel")


resizeW, resizeH = 320, 240
THRESHOLD_QUALITY_SSD = 11.75 #for SSD


def SSD_2_rectangles(detections, th):
    faces = []
    for i in range(detections[0][0].shape[0]):
      if(detections[0][0][i][2]>th):
        x1 = int(resizeW*detections[0][0][i][3])
        y1 = int(resizeH*detections[0][0][i][4])
        r1 = int(resizeW*detections[0][0][i][5])
        b1 = int(resizeH*detections[0][0][i][6])
        faces.append([x1,y1,r1-x1,b1-y1])
    return faces



def detectFace_SSD():

    capture = cv2.VideoCapture(dirVideo+"video.avi")

    #Create two opencv named windows
    nomb1 = "Main"
    cv2.namedWindow(nomb1, cv2.WINDOW_AUTOSIZE)

    #Position the windows next to each other
    cv2.moveWindow(nomb1,100,100)
    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)

    k=0
    try:
        while True:
            #Retrieve the latest image from the webcam
            rc,fullSizeBaseImage = capture.read()
            if not rc:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            baseImage = cv2.resize( fullSizeBaseImage, (resizeW, resizeH))

            pressedKey = cv2.waitKey(1)
            if pressedKey == ord('q') or  pressedKey == ord('Q'):
                cv2.destroyAllWindows()
                exit(0)

            imageBlob = cv2.dnn.blobFromImage(image = baseImage)

            detectorSSD.setInput(imageBlob)
            faces = SSD_2_rectangles(detectorSSD.forward(),0.70)

            #In the console we can show that only now we are
            #using the detector for a face
            print("Using the SSD detector to detect face",k)
            for (_x,_y,_w,_h) in faces:
               cv2.rectangle(baseImage, (_x,_y),(_x+_w,_y+_h),rectangleColor,2 )
               print( "Liveness module")
               print( "FER module")
            #Finally, we want to show the images on the screen
            cv2.imshow(nomb1, baseImage)

            k=k+1

    except KeyboardInterrupt as e:
        print(e)
        cv2.destroyAllWindows()
        exit(0)
