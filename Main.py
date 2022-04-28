from operator import invert
import cv2
from os import listdir
from os.path import isfile, join

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# now we have the haarcascades files 
# to detect the face and eyes to detect the face
face=cv2.CascadeClassifier("face.xml")

# to detect the eyes
eye=cv2.CascadeClassifier("eye.xml")



path = "./Fotos/"

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for i in onlyfiles:

    flagged = False

    # reading the webcam
    frame = cv2.imread(f"{path}{i}")

    scale_percent = 30 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # now the face is in the frame
    # the detection is done with the gray scale frame
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=1,
        minSize=(200, 250)
    )

    # now getting into the face and its position
    for (x,y,w,h) in sorted(faces, reverse=True, key=lambda i: (i[2], i[3]))[:1]:
        # drawing the rectangle on the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=2)

        # now the eyes are on the face
        # so we have to make the face frame gray
        gray_face=gray_frame[y:int(y+h/2),x:x+w]

        # make the color face also
        color_face=frame[y:int(y+h/2),x:x+w]

        # check the eyes on this face
        eyes=eye.detectMultiScale(
            gray_face,
            scaleFactor=1.3,
            minNeighbors=1,
            minSize=(40, 40),
            maxSize=(80, 80)
        )
 

        #get edges eyes
        s = sorted(eyes, key=lambda i: i[0])



        #edges
        if (len(s) >= 2):
            edges_eyes = [s[0], s[-1]]
            flagged = True
        
        # get into the eyes with its position
        for (a,b,c,d) in edges_eyes:
            # we have to draw the rectangle on the
            # coloured face
            cv2.rectangle(color_face,(a,b),(a+c,b+d),(0,255,0),thickness=2)

    if not flagged:
        continue

    cv2.imshow("Frame",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

"""
import cv2
from os import listdir
from os.path import isfile, join

path = "./Fotos/"

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for i in onlyfiles:
    image = cv2.imread(f"{path}{i}")

    scale_percent = 30 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(30, 30)
    )

    for (x,y,w,h) in eyes:
        cv2.rectangle(resized,(x,y),(x+w,y+h),(0, 255, 0),5)
    
    cv2.imshow("Eye Detected", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""