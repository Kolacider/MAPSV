import cv2 as cv
import numpy as np
#_*_ coding: utf-8 _*_ #


trained_car_data = cv.CascadeClassifier("vehicle_haarcascades.xml")

video = cv.VideoCapture("road10.mp4")

# 프레임사이즈 확인
'''frame_size = (int(video.get(cv.CAP_PROP_FRAME_WIDTH)),
              int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('original size: %d, %d' % frame_size)'''

RECT, RRECT = [0]*5, [0]*5
STRA = [0]*2
RET = [0]

# 차선감지 roi범위 지정,canny,blur 등 감지 알고리즘
def lanesDetection(img):
    height = img.shape[0]
    width = img.shape[1]

    region_of_interest_vertices = [(120, height-50), (width/2, 200), (520, height-50)]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    low_white = np.array([0, 0, 150])
    up_white = np.array([255, 75, 255])
    masked = cv.inRange(hsv, low_white, up_white)
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur_img = cv.GaussianBlur(masked,(9,9),0)
    edge_masked = cv.Canny(masked, 50, 100, apertureSize=3)
    edge_blurred = cv.Canny(blur_img, 50, 100, apertureSize=3)
    cropped_image = region_of_interest(
        edge_blurred, np.array([region_of_interest_vertices], np.int32))

    lines = cv.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
                           threshold=30, lines=np.array([]), minLineLength=10, maxLineGap=30)
    if (lines is None):
        lines = cv.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=0, lines=np.array([]),
                               minLineLength=10, maxLineGap=30)

    image_with_lines = draw_lines(img, lines)
    return image_with_lines, gray_img, edge_masked, cropped_image, masked, edge_blurred

# roi 설정
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    black = (255)
    cv.fillPoly(mask, vertices, black)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

# 검출된 선 그리기
def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    left_fit = []
    right_fit = []

    for line in lines:
        for x1,y1,x2,y2 in line:

            x1, y1, x2, y2 = line.reshape(4) # 라인의 끝점
            parameter = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameter[0] # 라인의 기울기
            intercept = parameter[1] # 라인의 y절편
            under_x = (360 - intercept) / slope

            c1 = int((230 - intercept) / slope), 230 # 라인의 230에서의 x값
            c2 = int((360 - intercept) / slope), 360 # 라인의 360에서의 x값
            # print(under_x)

            if (under_x < -2**20 or 280 < under_x <360 or under_x > 2**20):
                pass
                print("차선이탈", "%.2f" %under_x)

            if(int(abs(under_x)) < 10000):
                STRA[0]=c1
                STRA[1]=c2

        if(abs(slope)<3 and 0.55<abs(slope)):
            cv.line(blank_image, c1, c2, (255, 255, 0), 2)

        img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img



while True:
    ret, frame = video.read()
    # frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA) # 영상사이즈 조절 (코드는 640 X 360기준)

    if not ret:
        video = cv.VideoCapture("road10.mp4")
        continue

    xx, yy = 160, 200
    ww, hh = 320, 120

    subframe = frame[xx:xx + ww, yy:yy + hh]
    cv.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 255), 2)
    grayvid = cv.cvtColor(subframe, cv.COLOR_BGR2GRAY)
    vid3 = cv.GaussianBlur(grayvid,(11,11),0)

    carCoordinates = trained_car_data.detectMultiScale(vid3)

    for (x, y, w, h) in carCoordinates:

        if(y + yy > 240):
            p1 = int(x + 0.25 * w)
            p2 = int(y + 0.25 * h)
            p3 = int(x + 0.75 * w)
            p4 = int(y + 0.75 * h)
            pp1, pp2, pp3, pp4 = int(x +  w + xx), int(y + w + yy), int(x +  w + xx), int(y +  w + yy)
            cv.rectangle(subframe, (p1, p2), (p3, p4), (0, 0, 255), 2)


            gamma = (RECT[0] != p1)
            RECT[0], RECT[1], RECT[2], RECT[3], RECT[4] = p1, p2, p3, p4, (pp1,pp2,pp3,pp4)
            RRECT[0], RRECT[1], RRECT[2], RRECT[3] = pp1, pp2, pp3, pp4


        # 상자가 새로 떴을 때
            if(gamma == False):
            # 차의 빨간 박스와 검출된 line이 겹치는지 판단하는 retval
                retval, pt1, pt2 = cv.clipLine(RECT[4], STRA[0], STRA[1])
                # print(RECT[4], STRA[0], STRA[1])
            # print(retval)
                RET[0] = retval
            # print(retval, pt1, pt2)
                List_STRA1 = list(STRA[0])
                List_STRA2 = list(STRA[1])

                LN1 = [List_STRA1[0]-List_STRA2[0],List_STRA1[1]-List_STRA2[1]] # 직선 라인 and pp1
                LN2 = [RRECT[0]-List_STRA2[0],RRECT[1]-List_STRA2[1]] #
                LN3 = [RRECT[0] - List_STRA2[0], RRECT[2] - List_STRA2[1]]

                r1 = LN1[1]*LN2[0]-LN1[0]*LN2[1]
                r2 = LN1[1]*LN3[0]-LN1[0]*LN3[1]

                if(r1*r2<0):
                    print("충돌경보")
                else:
                    pass


    frame1, frame2, frame3, frame4, masked, frame5 = lanesDetection(frame)
    cv.imshow('image_with_lines', frame1)
    cv.imshow('cropped images', frame4)
    cv.imshow('masked', masked)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv.destroyAllWindows()
