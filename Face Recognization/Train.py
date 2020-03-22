import os
from PIL import Image
import numpy as np
import cv2
import pickle
cur_id=0
label_id={}
Train_ph=[]
Train_label=[]
base_dri='/home/vengadesh/Downloads/training'
face_cascade = cv2.CascadeClassifier('/home/vengadesh/PycharmProjects/sample/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
for root,dirs,files in os.walk(base_dri):
    for file in files:
        if file.endswith('JPG') or file.endswith('png') or file.endswith('jpg'):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path))
            print(label,path)
            if not label in label_id:
                label_id[label]=cur_id
                cur_id += 1
            id=label_id[label]
            print(label_id)

            pil_img=Image.open(path).convert("L")
            img_arr=np.asarray(pil_img,'uint8')
            print(img_arr)
            face=face_cascade.detectMultiScale(img_arr,1.3,5)
            for(x,y,w,h) in face:
                roi=img_arr[y:y+h, x:x+w]
                Train_ph.append(roi)
                Train_label.append(id)
with open("labels.pickle",'wb')as f:
    pickle.dump(label_id,f)
recognizer.train(Train_ph,np.array(Train_label))
recognizer.save("/home/vengadesh/Downloads/training/YML/train.yml")

