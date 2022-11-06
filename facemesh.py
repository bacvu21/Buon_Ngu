from sre_constants import SUCCESS
import cv2
import mediapipe as mp
import time


cap  =cv2.VideoCapture(0)
pTime =0
mpDraw =mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh


faceMesh = mpFaceMesh.FaceMesh(max_num_faces= 3)



while True:
    SUCCESS,img =cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    
    if result.multi_face_landmarks:
        
        for face_landmarks in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img,face_landmarks,mpFaceMesh.FACEMESH_CONTOURS)
            
    
    ctime =time.time()
    
    fps = 1/(ctime - pTime)
    pTime = ctime 
    cv2.putText(img ,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ('q'):
        break