'''
Project: 'Information Extraction from Citizen ID Card'

    Application specs:
        Major specs (must have)
            1. Get information from card as text using OCR.
            2. Get copied of card for certification.

        Minor spec(s) (may be in the future)
            1. Be capable in other cards. (ex. driving license card, passport)

    Environmental requirements:
        1. Background should be black or dark.
        2. Card's area should be the main object. (larger than 40% of the image)

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

''' ========== 1. Preprocess image ========== '''
# preprocess for getting edge from edge detection
resizedImage, grayImage, blurredImage, edgedCannyImage, dilatedImage, preProcessedImage = preprocess(
    originalImage, resizeWidth, resizeHeight, kSize)

# show all preprocessed images
showImages(
    [resizedImage, grayImage, blurredImage, edgedCannyImage, dilatedImage, preProcessedImage],
    ['RGB image', 'Grayscale image', 'Grayscale image with Gaussian filter',
    'Edge detection using Canny','Dilated image', 'Eroded image (final preprocess)'])


''' ========== 2. Specify location and area of card ========== '''
# @@@@@ 2.1 find contour of the card (assume that it is the biggest area in the image)
biggestContour = findBiggestContour(preProcessedImage)

# show the biggest contour in the image
biggestContourImage = resizedImage.copy()
cv2.drawContours(biggestContourImage, biggestContour, -1, (0, 255, 0), 3)
showImages([biggestContourImage], ['The biggest contour in the image'])


# @@@@@ 2.2. find 4 corners of the card as rectangle (from straight line of each side)
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


    ''' ========== 3. Crop image to get only card using warpPerspective ========== '''
    # @@@@@ 3.1. reorder corner points for applying warp prespective
    reOrderCornerPoints = reorderPoints(cornerPoints)

    # @@@@@ 3.2 define width and height of card image that prepare for warp
    cornerScale = originalImageWidth / resizeWidth
    oldPoints = np.float32(reOrderCornerPoints) * cornerScale

    newPoints = np.float32(
        [[0, 0], [cardWidth, 0], [0, cardHeight], [cardWidth, cardHeight]]
    )

    # @@@@@ 3.3 apply warpPerspective
    matrix = cv2.getPerspectiveTransform(oldPoints, newPoints)
    baseCardImage = cv2.warpPerspective(
        originalImage, 
        matrix, 
        (cardWidth, cardHeight)
    )

    showImages([baseCardImage], ['Only card'])


    ''' ========== 4. classify type of the card ========== '''
    # classify card in the function
    cardType, grayCardImage, templateMatchingImage = classifyCard(baseCardImage)

    # show matching value and location of detected logo
    showImages([templateMatchingImage], ['Detected logo location'])


    ''' ========== 5. extract information from card using OCR ========== '''
    binaryCardImage = cv2.adaptiveThreshold(
        grayCardImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
    )

    openBinaryCardImage = cv2.morphologyEx(binaryCardImage, cv2.MORPH_OPEN, np.ones((kSize, kSize)))
    closeBinaryCardImage = cv2.morphologyEx(openBinaryCardImage, cv2.MORPH_CLOSE, np.ones((kSize, kSize)))

    showImages(
        [binaryCardImage, openBinaryCardImage, closeBinaryCardImage],
        ['Binary card', 'Opened binary card from binary', 'Closed binary card from opened']
    )


    extractedInfos, headers, infoAreaImage = getInformationFromCard(
        baseCardImage, closeBinaryCardImage, cardType
    )

    showImages([infoAreaImage], ['Area of information'])

    for index in range(len(extractedInfos)):
        extractedInfos[index] = f'{headers[index]}: {extractedInfos[index]}'

    listToStr = '\n\n\n'.join(map(str, extractedInfos))

    text_file = open(r'text_output.txt','w', encoding='utf8') 
    text_file.write(listToStr)
    text_file.close()

    # show all results
    plt.show()

else:
    print('Cannot detect lines.')