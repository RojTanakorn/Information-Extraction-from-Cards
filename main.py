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
from constants import *


# Read card image as RGB image
originalImage = cv2.cvtColor(cv2.imread('./card_images/citizen_card1.jpg'), cv2.COLOR_BGR2RGB)

originalImageHeight, originalImageWidth = originalImage.shape[:-1]

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


# @@ if 4 lines are detected, continue processing
if lines is not None:
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
        originalImage, 
        matrix, 
        (cardWidth, cardHeight)
    )


    ''' ========== 4. classify type of the card ========== '''
    # @@@@@ 4.1 preprocess card image by converting to grayscale and apply filter
    grayCardImage = cv2.cvtColor(baseCardImage, cv2.COLOR_RGB2GRAY)
    grayCardImage = cv2.GaussianBlur(grayCardImage, (3, 3), 0)

    # @@@@@ 4.2 read logo images for classifying and store in list
    citizen_logo = cv2.imread('./card_logos/citizen_logo.jpg', cv2.IMREAD_GRAYSCALE)
    driving_logo = cv2.imread('./card_logos/driving_logo.jpg', cv2.IMREAD_GRAYSCALE)

    logo_list = [citizen_logo, driving_logo]
    maxVals = []

    for logo in logo_list:
        template = cv2.GaussianBlur(logo, (5, 5), 0)
        template = cv2.Canny(template, 75, 150)

        templateHeight, templateWidth = template.shape
        found = None

        for scale in np.linspace(0.2, 1.0, 20):
            card = cv2.resize(grayCardImage, (int(cardWidth * scale), int(cardHeight * scale)))
            r = grayCardImage.shape[1] / float(card.shape[1])

            if card.shape[0] < templateHeight or card.shape[1] < templateWidth:
                break

            canny = cv2.Canny(card, 25, 150)
            result = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        (maxVal, maxLoc, r) = found
        maxVals.append(maxVal)

        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + templateWidth) * r), int((maxLoc[1] + templateHeight) * r))

        cv2.rectangle(baseCardImage, (startX, startY), (endX, endY), (255, 0, 0), 2)


    ''' ========== 5. extract information from card using OCR ========== '''
    binaryCardImage = cv2.adaptiveThreshold(
        grayCardImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
    )

    binaryCardImage2 = cv2.morphologyEx(binaryCardImage, cv2.MORPH_OPEN, np.ones((kSize, kSize)))
    binaryCardImage3 = cv2.morphologyEx(binaryCardImage2, cv2.MORPH_CLOSE, np.ones((kSize, kSize)))


    infosDict = getInformationFromCard(binaryCardImage3) 
    # print(infosDict)

    # listToStr = '\n\n\n'.join(map(str, infosDict))

    # text_file = open(r'text_output.txt','w', encoding='utf8') 
    # text_file.write(listToStr)
    # text_file.close()



    ''' ********** Show all results ********** '''
    showedImages = [
        # originalImage,
        # resizedImage,
        # grayImage,
        # blurredImage,
        # edgedCannyImage,
        # dilatedImage,
        # preProcessedImage,
        # biggestContourImage,
        # blankImage,
        # fourLinesImage,
        # cardAreaImage,
        baseCardImage,
        # grayCardImage,
        # binaryCardImage,
        # binaryCardImage2,
        # binaryCardImage3,
        # canny,
        # template
    ]

    imageTitles = [
        'original image',
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
        'grayscale card',
        'binary card',
        # 'preprocess card'
    ]

    for i in range(len(showedImages)):
        plt.figure(imageTitles[i])
        if showedImages[i].ndim == 3:
            plt.imshow(showedImages[i])
        else:
            plt.imshow(showedImages[i], cmap='gray')
    plt.show()

else:
    print('Cannot detect lines.')