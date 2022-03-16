import dlib
import cv2 as cv
import numpy as np
from datetime import datetime
import json
from playsound import playsound as play

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

vid_in = cv.VideoCapture(1)

Average = []
Difference = []
Angular = []
Subtract_Right = [0]*2
Subtract_Left = [0]*2
temp = 0
i = 0

while True:
    ret, image_o = vid_in.read()
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

            A = np.array(landmark_list[36])
            B = np.array(landmark_list[39])
            distX = np.linalg.norm(A - B)
            C, D = np.array(landmark_list[37]), np.array(landmark_list[38])
            X = (C + D)/2
            E, F = np.array(landmark_list[40]), np.array(landmark_list[41])
            Y = (E + F)/2
            distY = np.linalg.norm(X - Y)
            Z = 0
            Z += distY
            K = np.array([Z])
            Ratio = 100*distY/distX




            if (Ratio >= 19):
                print("Ratio : ", Ratio)
                before1 = datetime.now()
            else:
                now1 = datetime.now()
                if (now1 - before1).seconds <= 1:
                    continue
                else:
                    print("Drowsiness Emergency")
                    play("Changu.mp3")

            # Head Angle
            Noseline1, Noseline2 = np.array(landmark_list[28]), np.array(landmark_list[29])
            Right_Lip, Right_Faceline = np.array(landmark_list[48]), np.array(landmark_list[4])
            Left_Lip, Left_Faceline = np.array(landmark_list[54]), np.array(landmark_list[12])
            Right_dist = np.linalg.norm(Right_Lip-Right_Faceline)
            Left_dist = np.linalg.norm(Left_Lip-Left_Faceline)
            differ_Right = abs(Right_dist / Left_dist)
            differ_Left = abs(Left_dist / Right_dist)

            if (len(face_detector) == 1) :
                i += 1
                Subtract_Right[int(i%2)] = differ_Right
                Sub_Right = abs(Subtract_Right[0] - Subtract_Right[1])
                Subtract_Left[int(i%2)] = differ_Left
                Sub_Left = abs(Subtract_Left[0] - Subtract_Right[1])

            J = np.array([0,1])
            L = np.array([0,1000])
            M = np.array([300,1])

            inner = np.dot(Noseline2-Noseline1,J)
            AB = np.linalg.norm(Noseline2-Noseline1) * np.linalg.norm(J)
            angle = np.arccos(inner / AB)

            Angular.insert(0,angle)

            if temp != distY :
                temp = distY
                Average.append(temp)
                AVG = sum(Average)/len(Average)
                Difference.append(AVG)


    if len(face_detector) :
        before2 = datetime.now()
    if not len(face_detector) and Angular[0] < 15 and Sub_Left < 1 :
        now2 = datetime.now()
        if (now2 - before2).seconds <= 2:
            continue
        else:
            print("Emergency")
            if (now2 - before2).seconds < 4:
                continue
            else:
                print("Auto Cruise Executed")

    else:
        before2 = datetime.now()

    cv.imshow('result', image)
    key = cv.waitKey(1)

    if key == 27:
        break
        play.terminate()

# plt.hist(Average)
# plt.show()

vid_in.release()

