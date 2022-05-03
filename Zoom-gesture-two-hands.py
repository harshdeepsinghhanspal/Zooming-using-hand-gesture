import cv2
from cvzone.HandTrackingModule import HandDetector
#use version 1.5.0 for cvzone
#mediapipe version 8.7.1

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
startDist = None
scale=0
cx, cy = 0, 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread('Resources/1.jpg')

    if len(hands)==2:
        #print(detector.fingersUp(hands[0]),detector.fingersUp(hands[1])) #Thumb and index
        if detector.fingersUp(hands[0])==[1,1,0,0,0] and detector.fingersUp(hands[1])==[1,1,0,0,0]:
            #[1,1,0,0,0]-Thumb and index up and rest down
            lmList1 = hands[0]['lmList']
            lmList2 = hands[1]['lmList']
            if startDist is None:
                length, info, img = detector.findDistance(lmList1[8],lmList2[8],img)
                startDist = length

            length, info, img = detector.findDistance(lmList1[8],lmList2[8],img)
            scale = int((length-startDist)//2) #sensitivity reduced
            cx,cy= info[4:]

    else:
        startDist = None #when fingers are closed and open, zero value
    try:
        h1, w1, _ = img1.shape
        newH, newW = ((h1+scale)//2)*2, ((w1+scale)//2)*2 #pixels can be odd values so to make it even
        img1 = cv2.resize(img1, (newW, newH))
        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1

    except:
        pass

    cv2.imshow('Result', img)
    cv2.waitKey(1)