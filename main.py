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
import matplotlib.pyplot as plt  # plot graph and show images
from utils import *  # all functions and variables for computing


''' ========== Constants ========== '''
resizeWidth = 800  # image width for resizing
resizeHeight = 600  # image height for resizing
kSize = 3  # size of kernel for dilation and erosion

cardWidth = 1376  # card width (8.6 x 160)
cardHeight = 864  # card height (5.4 x 160)

# Read card image as RGB image
originalImage = cv2.cvtColor(
    cv2.imread('./card_images/card4.jpg'), cv2.COLOR_BGR2RGB
)

originalImageHeight, originalImageWidth, _ = originalImage.shape

''' ========== 1. Preprocess image ========== '''
# preprocess for getting edge from edge detection
resizedImage, grayImage, blurredImage, edgedCannyImage, dilatedImage, preProcessedImage = preprocess1(
    originalImage, resizeWidth, resizeHeight, kSize)


''' ========== 2. Specify location and area of card ========== '''
# @@@@@ 2.1 find contour of the card (assume that it is the biggest area in the image)
biggestContour = findBiggestContour(preProcessedImage)

# show the biggest contour in the image
biggestContourImage = resizedImage.copy()
cv2.drawContours(biggestContourImage, biggestContour, -1, (0, 255, 0), 3)


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
cornerPoints, fourLinesImage = findFourCorners(resizedImage, lines)

# show 4 corners in the image
cardAreaImage = resizedImage.copy()
for pt in cornerPoints:
    cv2.circle(cardAreaImage, tuple(pt), 5, (0, 255, 0), -1)


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
    originalImage, matrix, (cardWidth, cardHeight))


''' ********** Show all results ********** '''
showedImages = [originalImage,
                resizedImage,
                grayImage,
                blurredImage,
                edgedCannyImage,
                dilatedImage,
                preProcessedImage,
                biggestContourImage,
                blankImage,
                fourLinesImage,
                cardAreaImage,
                baseCardImage,
                # grayCardImage,
                # adaptiveThImage
                ]

imageTitles = ['original image',
               'resized image',
               'grayscale image',
               'blurred image',
               'edge detection using Canny',
               'dilated image',
               'final preprocessed image (after erosion)',
               'the biggest contour',
               'contour in blackground',
               'line on sides of card',
               'corner of card',
               'base card',
               #    'grayscale card',
               #    'binary card'
               ]

for i in range(len(showedImages)):
    plt.figure(imageTitles[i])
    if showedImages[i].ndim == 3:
        plt.imshow(showedImages[i])
    else:
        plt.imshow(showedImages[i], cmap='gray')
plt.show()