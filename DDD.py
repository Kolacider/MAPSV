import dlib
import cv2 as cv
import json
import numpy as np
from datetime import datetime
from socket import *
from playsound import playsound as play

port = 1000
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind(('', port))
serverSock.listen(1)

print('%d번 포트로 접속 대기중...'%port)
connectionSock, addr = serverSock.accept()
print(str(addr), '에서 접속되었습니다.')

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INLINE = list(range(61, 68))
JAWLINE = list(range(0, 17))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

video = cv.VideoCapture(0)

Angular = [0]
T_Right = [0]*3
T_Left = [0]*3
i = 0

while True:
    ret, image_o = video.read()
    image = cv.resize(image_o, dsize=(640, 480), interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = detector(img_gray, 1)

    for face in face_detector:

        landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기
        landmark_list = []

        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)

        with open("test.json", "w") as json_file:
            key_val = [ALL, landmark_list]
            landmark_dict = dict(zip(*key_val))
            json_file.write(json.dumps(landmark_dict))
            json_file.write('\n')

            A_Left, B_Left = np.array(landmark_list[36]), np.array(landmark_list[39])
            Width_Left = np.linalg.norm(A_Left - B_Left)
            C_Left, D_Left = np.array(landmark_list[37]), np.array(landmark_list[38])
            E_Left, F_Left = np.array(landmark_list[40]), np.array(landmark_list[41])
            X_Left, Y_Left = (C_Left + D_Left)/2, (E_Left + F_Left)/2
            Height_Left = np.linalg.norm(X_Left - Y_Left)

            A_Right, B_Right = np.array(landmark_list[42]), np.array(landmark_list[45])
            Width_Right = np.linalg.norm(A_Right - B_Right)
            C_Right, D_Right = np.array(landmark_list[43]), np.array(landmark_list[44])
            E_Right, F_Right = np.array(landmark_list[47]), np.array(landmark_list[46])
            X_Right, Y_Right = (C_Right + D_Right)/2, (E_Right + F_Right)/2
            Height_Right = np.linalg.norm(X_Right - Y_Right)
            L_Ratio = 100*Height_Left/Width_Left
            R_Ratio = 100*Height_Right/Width_Right

            if (L_Ratio >= 19 and R_Ratio >=19):
                before1 = datetime.now()
            else:
                now1 = datetime.now()

                if (now1 - before1).seconds <= 1:
                    continue
                else:
                    play("Beepsound1.mp3")

            # Head Angle
            Nose1, Nose2 = np.array(landmark_list[28]), np.array(landmark_list[29])
            Lip_Left, Faceline_Left = np.array(landmark_list[48]), np.array(landmark_list[4])
            Lip_Right, Faceline_Right = np.array(landmark_list[54]), np.array(landmark_list[12])
            Noseline = Nose1 - Nose2
            Cheek_Left = np.linalg.norm(Lip_Left - Faceline_Left)
            Cheek_Right = np.linalg.norm(Lip_Right - Faceline_Right)
            Turn_Left = abs(Cheek_Left / Cheek_Right) # 왼쪽 사이드미러 볼 때 커짐
            Turn_Right = abs(Cheek_Right / Cheek_Left) # 오른쪽 사이드미러 볼 때 커짐


            if (len(face_detector) == 1) :
                i += 1
                T_Right[int(i%3)] = Turn_Right
                T_Left[int(i%3)] = Turn_Left

            TT_Right = np.mean(T_Right)/np.mean(T_Left)
            TT_Left = np.mean(T_Left)/np.mean(T_Right)
            J = np.array([0,1])
            inner = np.dot(Noseline,J)
            AB = np.linalg.norm(Noseline) * np.linalg.norm(J)
            angle = abs(180-np.arccos(inner / AB)*180/np.pi)
            Angular.insert(0,angle)


    if len(face_detector) :
        before2 = datetime.now()

    if not len(face_detector) and Angular[0] < 15 and TT_Left < 5 and TT_Right < 5:
        now2 = datetime.now()

        if (now2 - before2).seconds <= 2:
            continue
        else:
            play("Beepsound2.mp3")
            if (now2 - before2).seconds < 4:
                continue
            else:
                play("Beepsound3.mp3")
                sendData = 'BCW'
                connectionSock.send(sendData.encode('utf-8'))
                serverSock.close()
                break

    else:
        before2 = datetime.now()

    cv.imshow('result', image)
    key = cv.waitKey(1)

    if key == 27:
        break

video.release()
cv.destroyAllWindows()
