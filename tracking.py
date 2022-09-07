import cv2

cap = cv2.VideoCapture('example.mp4')

#sabit bir kameradan elde edilen görüntüden hareket eden onjeleri alır
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# roi kullanılmadı, kamera görüntüsünde yeterince spesifik bir bölüm alınmış

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    # print(height,width)
    # roi = frame[0:360,150:620]
    mask = object_detector.apply(frame) #frameden hareket eden nesneleri alacak
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #254=beyaz sadece bunları tut, gölgeyi sil
    # cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ =cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: #1000 pikselden büyükse kontür çiz değilse çizme
            #cv2.drawContours(roi, [cnt], -1, (0,255, 0), 2)
             x, y, w, h = cv2.boundingRect(cnt)
             cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),3)

    # cv2.imshow("roi", roi)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    if cv2.waitKey(1)== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

