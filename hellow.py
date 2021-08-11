import cv2 as cv


face_c = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_c = cv.CascadeClassifier('haarcascade_eye.xml')
def detect(gray,frame):
    face = face_c.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eyes_c.detectMultiScale(roi_gray,1.1,25)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame

video_capture = cv.VideoCapture(0)

while True:
    _,frame = video_capture.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv.imshow('Video',canvas)
    k = cv.waitKey(1)

    if k == 27:
        cv.destroyAllWindows()
    elif k == ord('s'):
        video_capture.release()
        cv.destroyAllWindows()


