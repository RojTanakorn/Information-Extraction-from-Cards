'''
Project: 'Information Extraction from Official Cards'

    Application specs:
        1. Get information from the card as text using OCR (citizen card and driving license card).
        2. Get copied of the card for certification.

    Environmental requirements:
        1. Background should be smooth and dark/contrast.
        2. Card's area should be the main object. (larger than 40% of the image)

    Libraries:
        1. OpenCV - 4.4.0
        2. Numpy - 1.19.3
        3. Matplotlib - 3.3.3
        4. Pytesseract - 0.3.6
'''

# Import libraries
import cv2  # provides image processing tools
import numpy as np  # deal with image array
from utils import *  # all functions and variables for computing
from constants import *


# read card image as RGB image
originalImage = cv2.cvtColor(cv2.imread('./card_images/citizen_card1.jpg'), cv2.COLOR_BGR2RGB)

# show original image
showImages([originalImage], ['Original image as default size'])

originalImageHeight, originalImageWidth = originalImage.shape[:-1]

''' ========== 1.1 Preprocess image ========== '''
# preprocess for getting edge from edge detection
resizedImage, grayImage, blurredImage, edgedCannyImage, dilatedImage, preProcessedImage = preprocess(
    originalImage, resizeWidth, resizeHeight, kSize)

# show all preprocessed images
showImages(
    [resizedImage, grayImage, blurredImage, edgedCannyImage, dilatedImage, preProcessedImage],
    ['RGB image', 'Grayscale image', 'Grayscale image with Gaussian filter',
    'Edge detection using Canny','Dilated image', 'Eroded image (final preprocess)'])


''' ========== 1.2 Specify location and area of card ========== '''
# @@@@@ 1. find contour of the card (assume that it is the biggest area in the image)
biggestContour = findBiggestContour(preProcessedImage)

# show the biggest contour in the image
biggestContourImage = resizedImage.copy()
cv2.drawContours(biggestContourImage, biggestContour, -1, (0, 255, 0), 3)
showImages([biggestContourImage], ['The biggest contour in the image'])


# @@@@@ 2. find 4 corners of the card as rectangle (from straight line of each side)
# @@@@@ Note: card's shape is round rectangle, but corners should be rectangle

# @@ first, draw card's contour in blank background (black)
# @@ then apply erosion for finding straight lines efficiently
contourInBlackImage = np.zeros((resizeHeight, resizeWidth), dtype=np.uint8)
cv2.drawContours(contourInBlackImage, biggestContour, -1, (255, 255, 255), 3)
contourInBlackImage = cv2.erode(contourInBlackImage, np.ones((kSize+1, kSize+1)), iterations=1)
showImages([contourInBlackImage], ['Contour in black background with applying erosion'])

# @@ second, find 4 straight lines, which are each side of the card
lines = findLineFromContour(contourInBlackImage)

# @@ if 4 lines are detected, continue processing
if lines is not None:
    # @@ third, compute intersection points from 4 lines, get 4 points
    # @@  note: cornerPoints - 4 coordinates of the corner
    # @@  note: fourLinesImage - image that represent location of 4 lines
    cornerPoints, fourLinesImage = findFourCorners(resizedImage, lines)
    showImages([fourLinesImage], ['4 lines of each side of the card'])

    # show 4 corners in the image
    cardCornerImage = resizedImage.copy()
    for pt in cornerPoints:
        cv2.circle(cardCornerImage, tuple(pt), 5, (0, 255, 0), -1)
    showImages([cardCornerImage], ['4 corners of the card'])


    ''' ========== 1.3 Crop image to get only card using warpPerspective ========== '''
    # @@@@@ 1. reorder corner points for applying warp prespective
    reOrderCornerPoints = reorderPoints(cornerPoints)

    # @@@@@ 2. define width and height of card image that prepare for warp
    cornerScale = originalImageWidth / resizeWidth
    oldPoints = np.float32(reOrderCornerPoints) * cornerScale

    newPoints = np.float32(
        [[0, 0], [cardWidth, 0], [0, cardHeight], [cardWidth, cardHeight]]
    )

    # @@@@@ 3. apply warpPerspective
    matrix = cv2.getPerspectiveTransform(oldPoints, newPoints)
    baseCardImage = cv2.warpPerspective(
        originalImage, 
        matrix, 
        (cardWidth, cardHeight)
    )

    showImages([baseCardImage], ['Only card'])


    ''' ========== 1.4 classify type of the card ========== '''
    # classify card in the function
    cardType, grayCardImage, templateMatchingImage = classifyCard(baseCardImage)

    # show matching value and location of detected logo
    showImages([templateMatchingImage], ['Detected logo location'])


    ''' ========== 1.5 extract information from card using OCR ========== '''
    # apply Adaptive thresholding
    binaryCardImage = cv2.adaptiveThreshold(
        grayCardImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
    )

    # apply opening
    openBinaryCardImage = cv2.morphologyEx(binaryCardImage, cv2.MORPH_OPEN, np.ones((kSize, kSize)))

    # apply closing
    closeBinaryCardImage = cv2.morphologyEx(openBinaryCardImage, cv2.MORPH_CLOSE, np.ones((kSize, kSize)))

    # show results
    showImages(
        [binaryCardImage, openBinaryCardImage, closeBinaryCardImage],
        ['Binary card', 'Opened binary card from binary', 'Closed binary card from opened']
    )

    # get information from card using Pytesseract (OCR)
    extractedInfos, headers, infoAreaImage = getInformationFromCard(
        baseCardImage, closeBinaryCardImage, cardType
    )

    # show area of information
    showImages([infoAreaImage], ['Area of information'])

    # modify string information for representation
    for index in range(len(extractedInfos)):
        extractedInfos[index] = f'{headers[index]}: {extractedInfos[index]}'

    listToStr = '\n\n\n'.join(map(str, extractedInfos))

    # write all information to text_output.txt file for representation
    text_file = open(r'text_output.txt','w', encoding='utf8') 
    text_file.write(listToStr)
    text_file.close()

    # show all results
    plt.show()


    ''' ========== 2. A4: Copied of card ========== '''
    # width and height of A4 on 300 dpi
    a4Height = 3508
    a4Width = 2480

    # create white A4 plane
    a4 = (np.ones((a4Height, a4Width), dtype=np.uint8)) * 255

    # calculate center point of A4
    centerA4 = (int(a4Height/2), int(a4Width/2))

    # calculate start point of card on A4
    startA4 = (centerA4[0] - int(cardHeight/2), centerA4[1] - int(cardWidth/2))

    # put grayscale card image to A4 at the center
    a4[startA4[0]:startA4[0]+cardHeight, startA4[1]:startA4[1]+cardWidth] = grayCardImage

    # show A4 result
    showImages([a4], ['A4 of copied card'])
    plt.show()
else:
    print('Cannot detect lines.')