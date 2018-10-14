# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detectPlatesInScene(imgOriginalScene):

    listOfPossiblePlates = []

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    #grayscale and threshold images preprocessing
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)

    # find all possible chars in the scene,
    # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
    # Here we get a list of all the contours in the image that may be characters.
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)





    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)



    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)

        if possiblePlate.imgPlate is not None:
            listOfPossiblePlates.append(possiblePlate)


    return listOfPossiblePlates


def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    # find all contours
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):

        # Here we calculate the x,y,w,h,flatdiagonalsize,aspedctratio,area and (x,y) of the center of the rectangle that is bounding the contour.
        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):
            intCountOfPossibleChars = intCountOfPossibleChars + 1
            listOfPossibleChars.append(possibleChar)
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)


    return listOfPossibleChars


def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()
    # sort chars from left to right based on x position
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0
    # This is the probable central point of this plate.
    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
    # Here we calculate the probable width of this plate.
    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # Here we calculate the probale height of this particular plate.
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
    # We include the padding factor.
    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    # The first point is the point of rotation or center,theta and scaling factor
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    # unpack original image width and height
    height, width, numChannels = imgOriginal.shape
    # rotate the entire image
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))
    # We extract the probable plate from the Original image
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
    # copy the cropped plate image into the applicable member variable of the possible plate
    possiblePlate.imgPlate = imgCropped

    return possiblePlate
