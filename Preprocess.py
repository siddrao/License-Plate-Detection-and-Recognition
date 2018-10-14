# Preprocess.py

import cv2
import numpy as np
import math
import Main
import os
# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

###################################################################################################
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal) # We get the gray scale of the image.
    #cv2.imshow(" Grayscale", imgGrayscale)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'Grayscale.jpg'),imgGrayscale)
    # cv2.waitKey(0)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    #cv2.imshow('MaxContrastGrayscale', imgMaxContrastGrayscale)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'MaxContrastGrayscale.jpg'),imgMaxContrastGrayscale)
    # cv2.waitKey(0)

    height,width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    # 2nd parameter is (height,width) of Gaussian kernel,3rd parameter is sigmaX,4th parameter is sigmaY(as not specified it is made same as sigmaX).
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # cv2.imshow('GaussianBlur', imgBlurred)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'GaussianBlur.jpg'),
                imgBlurred)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    # cv2.imshow('adaptiveThreshold', imgThresh)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'adaptiveThreshold.jpg'),
                imgThresh)
    cv2.destroyAllWindows()

    return imgGrayscale, imgThresh

def preprocess1(imgOriginal):
    imgGrayscale = extractValue(imgOriginal) # We get the gray scale of the image.
    #cv2.imshow(" Grayscale", imgGrayscale)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'Grayscale1.jpg'),imgGrayscale)
    # cv2.waitKey(0)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    #cv2.imshow('MaxContrastGrayscale', imgMaxContrastGrayscale)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'MaxContrastGrayscale1.jpg'),imgMaxContrastGrayscale)
    # cv2.waitKey(0)

    height,width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    # 2nd parameter is (height,width) of Gaussian kernel,3rd parameter is sigmaX,4th parameter is sigmaY(as not specified it is made same as sigmaX).
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # cv2.imshow('GaussianBlur', imgBlurred)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'GaussianBlur1.jpg'),
                imgBlurred)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    # cv2.imshow('adaptiveThreshold', imgThresh)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join('C:/Users/Siddhant Rao/Desktop/ALPR-master/Main Program/output', 'adaptiveThreshold1.jpg'),
                imgThresh)
    cv2.destroyAllWindows()

    return imgGrayscale, imgThresh


def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue


def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    # Same as np.ones((3,3)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # It is difference of  input image and Opening of the image
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    # it is difference of closing of the input image and input image.
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
