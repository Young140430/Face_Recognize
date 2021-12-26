import cv2

cap = cv2.VideoCapture("E:/face_video/face_v/7/3.mp4")
x=1
while True:
    ret,frame = cap.read()
    if ret:
        cv2.imshow("video",frame)
        if cv2.waitKey(1000//24) & 0xff==ord("q"):
            break
        cv2.imwrite("F:/face_data_test/{0}-{1}.jpg".format(1,x),frame)
        x+=1
    else:
        break
cap.release()
cv2.destroyAllWindows()
