import cv2
from os import listdir
from os.path import isfile, join

nose_cascade = cv2.CascadeClassifier('./nose.xml')

path = "./Fotos/"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

ds_factor = 0.4

skipped = 0
found = 0

for i in onlyfiles:

    frame = cv2.imread(f"{path}{i}")

    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nose_rects = nose_cascade.detectMultiScale(
        gray[400:700, 250:600],
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(85, 85)
    )

    #choose correct one
    sorted_noses = sorted(nose_rects, key=lambda element: element[0])

    if not len(sorted_noses):
        skipped += 1
        continue
    else:
        found += 1

    #choose the middle one
    middle_index = (len(sorted_noses) // 2)

    #get coordsfor (x,y,w,h) in nose_rects:
    (x,y,w,h) = sorted_noses[middle_index-1]
    
    #draw rectangle
    """
    cv2.rectangle(frame, (x+250,y+400), (x+250+w,y+400+h), (0,255,0), 3)

    cv2.imshow(f"skipped: {skipped}; found: {onlyfiles.index(i)-skipped}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    print(f"skipped: {skipped}; found: {onlyfiles.index(i)-skipped}")