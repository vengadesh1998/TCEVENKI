import cv2
cam=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/vengadesh/Testing/testhead.xml')
eye_cascade = cv2.CascadeClassifier('/home/vengadesh/Testing/testeye.xml')
while 1:
    C,S=cam.read()
    grey=cv2.cvtColor(S,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(grey,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.rectangle(grey, (x, y), (x + w, y + h), (255, 255, 0), 2)
        r_grey = grey[y:y + h, x:x + w]
        r_color = S[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(r_grey)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(r_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)
    cv2.imshow('demo',S)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break

cam.release()
cv2.destroyAllWindows()



