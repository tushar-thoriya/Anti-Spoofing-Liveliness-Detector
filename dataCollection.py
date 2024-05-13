from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time

#################################
classID = 1   # 0 -> fake and 1 -> real
confidence = 0.8
save = True
bluerThreshold = 35 # Large is more focus 
debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatignPoint = 6
outputFolderPath = 'Dataset/DataCollect'
#########################


cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
# Run the loop to continually get frames from the webcam
while True:
    success, img = cap.read()
    img_Orig = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)
    
    listBlur = [] # True False values indicating if the faces are blur or not
    listInfo = [] # The normalized values and the class name for the label txt file
    
    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            # print(x, y, w, h)
            
            ##### Check the socre 
            score = int(bbox['score'][0] * 100)
            
            if score > confidence:
                #### Adding an offset tho the face Detected -----
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH*3)
                h = int(h + offsetH*3.5)
                
                #### To avoid vlues below 1 ------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0
                
                ### Find Blurriness --------------
                imgFace = img[y:y+h, x:x+w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > bluerThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)
                
                #### Normalize Values --------------
                ih, iw, _ = img.shape
                xc, yc = x + w/2, y + h/2
                xcn, ycn = round(xc / iw, floatignPoint), round(yc / ih, floatignPoint)
                wn, hn = round(w / iw, floatignPoint), round(h / ih, floatignPoint)
                # print(xcn, ycn, wn, hn)
                
                #### To avoid vlues above 1 ------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1
                
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")  # yolo requried format
                
                
                # cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
            
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(img, f'Score: {score}% Blur: {blurValue}', (x, y - 10), scale=2, thickness=3)
                cvzone.cornerRect(img, (x, y, w, h))
                
                if debug:
                    cv2.circle(img_Orig, center, 5, (255, 0, 255), cv2.FILLED)
                    cvzone.putTextRect(img_Orig, f'Score: {score}% Blur: {blurValue}', (x, y - 10), scale=2, thickness=3)
                    cvzone.cornerRect(img_Orig, (x, y, w, h))

        ### To Save ---------------
        if save:
            if all(listBlur) and listBlur != []:
                print(listBlur)
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img_Orig)  # give name as current second (23.64 = '23' + '64' = 2364)
                ### -------------- Save Label Text File -----------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)