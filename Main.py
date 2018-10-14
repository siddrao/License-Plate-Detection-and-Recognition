# Main.py

import cv2
import numpy as np
import os
import time
import DetectChars
import DetectPlates
import PossiblePlate

# module level variables #
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def main(image):
    CnnClassifier = DetectChars.loadCNNClassifier()  # load CNN

    if CnnClassifier == False:  # if CNN fails
        print("\nerror: CNN traning was not successful\n")  # error
        return

    imgOriginalScene = cv2.imread(image)  # open image
    #cv2.imshow(" Original image", imgOriginalScene)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'original-img.jpg'),imgOriginalScene)
    cv2.waitKey(0)

    imgOriginalScene = cv2.resize(imgOriginalScene, (0, 0), fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Resized Original image", imgOriginalScene)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'resize-img.jpg'),imgOriginalScene)

    if imgOriginalScene is None:
        print("\nerror: image not read from file \n\n")
        os.system("pause")
        return
    # combinations of contours that may be a plate.
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

    # detect chars in plates
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    if len(listOfPossiblePlates) == 0:  # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found
        response = ' '
        return response, imgOriginalScene
    else:


        # sort  from most number of chars to least number of chars)
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

        # if 1st plate is the actual plate
        licPlate = listOfPossiblePlates[0]


        if len(licPlate.strChars) == 0:
            print("\nno characters were detected\n\n")
            return ' ', imgOriginalScene


        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        print("\nlicense plate read from ", image, " :", licPlate.strChars, "\n")
        print("----------------------------------------")

        #
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        #

        cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'Detected-img.jpg'),
                    imgOriginalScene)
        #cv2.waitKey(0)

    return licPlate.strChars, licPlate.imgPlate



def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    #bounding rectangle
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)

    # get rect co-ordinates
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):  # if  plate in upper three quarter of the image
        # display chars below plate
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:  #  if  plate in lower quarter of the image
        # display chars above plate
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))


    textSizeWidth, textSizeHeight = textSize
    # calculate the lower left origin,center width,height of the text area
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))
    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_GREEN, intFontThickness)


if __name__ == "__main__":
    #main('OS269DT.jpg')
    main('Test_car_images_dataset/785K686.jpg')
