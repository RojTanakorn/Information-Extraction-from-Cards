import cv2  # provides image processing tools
import numpy as np  # deal with image array
import matplotlib.pyplot as plt  # plot graph and show images
from utils import *  # all functions and variables for computing
from constants import *


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
address = 'https://192.168.1.7:3000/video'
camera.open(address)

isCapture = False

while True:
    ret, originalImage = camera.read()
    originalImage = cv2.resize(originalImage, (800, 600))

    originalImageHeight, originalImageWidth = originalImage.shape[:-1]

    ''' ========== 1. Preprocess image ========== '''
    # preprocess for getting edge from edge detection
    resizedImage, grayImage, blurredImage, edgedCannyImage, dilatedImage, preProcessedImage = preprocess(
        originalImage, resizeWidth, resizeHeight, kSize)

    ''' ========== 2. Specify location and area of card ========== '''
    # @@@@@ 2.1 find contour of the card (assume that it is the biggest area in the image)
    biggestContour = findBiggestContour(preProcessedImage)

    # show the biggest contour in the image
    # biggestContourImage = resizedImage.copy()
    # cv2.drawContours(biggestContourImage, biggestContour, -1, (0, 255, 0), 3)


    # @@@@@ 2.2. find 4 corners of the card as rectangle (from straight line of each side)
    # @@@@@ Note: card's shape is round rectangle, but corners should be rectangle

    # @@ first, draw card's contour in blank background (black)
    # @@ then apply erosion for finding straight lines efficiently
    blankImage = np.zeros((resizeHeight, resizeWidth), dtype=np.uint8)
    cv2.drawContours(blankImage, biggestContour, -1, (255, 255, 255), 3)
    blankImage = cv2.erode(blankImage, np.ones((kSize+1, kSize+1)), iterations=1)

    # @@ second, find 4 straight lines, which are each side of the card
    lines = findLineFromContour(blankImage)

    # @@ third, compute intersection points from 4 lines, get 4 points
    # @@  note: cornerPoints - 4 coordinates of the corner
    # @@  note: fourLinesImage - image that represent location of 4 lines
    if lines is not None:
        cornerPoints, fourLinesImage = findFourCorners(resizedImage, lines)
        cv2.imshow('camera', fourLinesImage)
    else:
        cv2.imshow('camera', originalImage)

    # cv2.imshow('camera', originalImage)
    # if(cv2.waitKey(1) & 0xFF==ord('q')):
    #     break
    if(cv2.waitKey(1) & 0xFF==ord('c')):
        isCapture = True
        captureImage = originalImage
        captureImageWithLines = fourLinesImage
        break

if isCapture:
    camera.release()
    cv2.destroyAllWindows()
    concatImage = cv2.resize(cv2.hconcat((captureImage, captureImageWithLines)), (1200, 450))
    cv2.imshow('capture', concatImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()