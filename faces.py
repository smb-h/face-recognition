import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


labels = {}
# DeSerialize label id's
with open("labels.pickle", 'rb') as f:
    org_labels = pickle.load(f)
    labels = {v:k for k,v in org_labels.items()}


cap = cv2.VideoCapture(0)

stroke = 2

img_item = "my-image.png"
rec_stroke = 2
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        font = cv2.FONT_HERSHEY_SIMPLEX
        white_color = (255, 255, 255)
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 85:
            name = labels[id_]
            cv2.putText(frame, name, (x, y), font, 1, white_color, stroke, cv2.LINE_AA)

        cv2.imwrite(img_item, roi_color)

        # Draw rectangle
        rec_color = (255, 30, 30) # BGR as blue green red
        cv2.rectangle(frame, (x, y), (x+w, y+h), rec_color, rec_stroke)
        # detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh)in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # detect smile
            # smiles = smile_cascade.detectMultiScale(roi_gray)
            # for (sx, sy, sw, sh)in smiles:
            #     cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)



    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
