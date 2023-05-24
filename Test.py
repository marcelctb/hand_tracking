import cv2
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
import numpy as np
import math
from keras.preprocessing import image

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = load_model('Model/imageclassifier.h5')

offset = 20
imgSize = 300

folder = "Data/com_marcas/training/C"
counter = 0

classes = 3
letras = {'0' : 'A', '1' : 'B', '2' : 'C'}
maior, class_index = 0, 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

                imgTest = cv2.resize(imgWhite, (64, 64))
                imgTest = image.image_utils.img_to_array(imgTest)
                imgTest = np.expand_dims(imgTest, axis=0)
                result = classifier.predict(imgTest)

                maior, class_index = -1, -1

                for z in range(classes):
                    if result[0][z] > maior:
                        maior = result[0][z]
                        class_index = z

                print(result, letras[str(class_index)])
            except:
                print('Erro na captura')
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

                imgTest = cv2.resize(imgWhite, (64, 64))
                imgTest = image.image_utils.img_to_array(imgTest)
                imgTest = np.expand_dims(imgTest, axis=0)
                result = classifier.predict(imgTest)

                maior, class_index = -1, -1

                for z in range(classes):
                    if result[0][z] > maior:
                        maior = result[0][z]
                        class_index = z

                print(result, letras[str(class_index)])
            except:
                print('Erro na captura')

        cv2.putText(imgOutput, letras[str(class_index)], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        try:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        except:
            print('Erro na captura')

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break