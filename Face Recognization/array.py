
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('/home/vengadesh/PycharmProjects/sample/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/home/vengadesh/Downloads/training/YML/train.yml")

eye_cascade = cv2.CascadeClassifier('/home/vengadesh/PycharmProjects/sample/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

with open("labels.pickle",'rb')as f:
    o_labels=pickle.load(f)
    print(o_labels)
    labels={v:k for k,v in o_labels.items()}
    print(labels)



cap = cv2.VideoCapture(0)


while 1:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        id,conf=recognizer.predict(roi_gray)
        if ((conf>45) and (conf<85)):
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id]
            color=(255,255,255)
            stroke=2
            cv2.putText(img,name,(x,y),font,1,color,stroke,cv2.LINE_AA)






    # Display an image in a window
    cv2.imshow('img',img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
