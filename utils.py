# Import libraries
import cv2  # provides image processing tools
import numpy as np  # deal with image array
import pytesseract # provides OCR

''' Constants '''
# location of information in the card (from 688x432)
citizenCardArea = {
    'id': {'start': (606, 92), 'end': (1085, 154)},
    'name': {'start': (389, 156), 'end': (1350, 256)},
    'dateOfBirth': {'start': (603, 378), 'end': (900, 442)},
    'address': {'start': (137, 553), 'end': (1020, 686)}
}



def preprocess1(originalImage, imageWidth, imageHeight, kSize):
    # Resize image for processing
    resizedImage = cv2.resize(originalImage, (imageWidth, imageHeight))

    # Convert RGB image to grayscale image
    grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_RGB2GRAY)

    # Blur grayscale image with Gaussian for reducing noise
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

    # Apply Canny edge detection for getting edge in grayscale image
    edgedCannyImage = cv2.Canny(blurredImage, 75, 150)

    # Apply dilation for connecting some broken edges
    dilatedImage = cv2.dilate(
        edgedCannyImage, np.ones((kSize, kSize)), iterations=2
    )

    # Apply erosion for reducing size of edge line
    erodedImage = cv2.erode(
        dilatedImage, np.ones((kSize, kSize)), iterations=1
    )

    return resizedImage, grayImage, blurredImage, edgedCannyImage, dilatedImage, erodedImage


def findBiggestContour(preProcessedImage):
    # find objects area using contour from edged image
    contours, _ = cv2.findContours(
        preProcessedImage,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # find contour of card (assume that it is the biggest area)
    biggestContour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    return biggestContour


def findLineFromContour(blankImage):
    lines = np.array([])
    lineThreshold = 250  # threshold of line (minimum length)

    # apply Hough Transform algorithm for getting lines
    # use while loop for adjusting threshold that we can get 4 lines
    while len(lines) != 4:
        lines = cv2.HoughLines(
            blankImage, 1, np.pi / 180, lineThreshold, None, 0, 0
        )
        lineThreshold = lineThreshold+5 if len(lines) > 4 else lineThreshold-5

    return lines


def findFourCorners(resizedImage, lines):
    hLineList = []  # list of horizontal lines (contain 2 points of line)
    vLineList = []  # list of vertical lines (contain 2 points of line)

    # list of angles for making decision of line type (0, 90, 180 degrees)
    angleList = np.array([0., np.pi/2, np.pi])

    # image that represent location of 4 lines
    fourLinesImage = resizedImage.copy()

    # compute 2 points and classify type of the line
    for line in lines:
        rho, theta = line[0][0], line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        angleDiffList = np.abs(angleList - theta)
        minIndex = np.argmin(angleDiffList)

        if minIndex == 1:
            vLineList.append([pt1, pt2])
        else:
            hLineList.append([pt1, pt2])

        cv2.line(fourLinesImage, pt1, pt2, (0, 255, 0), 3)

    # list that contain 4 intersection points
    cornerPoints = []

    # compute intersection points by formulas
    # ref: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    for h in range(len(hLineList)):
        x1, y1 = hLineList[h][0]
        x2, y2 = hLineList[h][1]
        for v in range(len(vLineList)):
            x3, y3 = vLineList[v][0]
            x4, y4 = vLineList[v][1]

            intersectX = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)
                          ) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            intersectY = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)
                          ) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            cornerPoints.append([round(intersectX), round(intersectY)])

    cornerPoints = np.array(cornerPoints)

    return cornerPoints, fourLinesImage


def reorderPoints(cornerPoints):
    # sort from x value for classifying left side and right side
    cornerPoints = cornerPoints[cornerPoints[:, 0].argsort()]

    reOrderCornerPoints = np.zeros((4, 2), dtype=np.int32)

    # first 2 points are points at the left side of the card
    arrLeft = cornerPoints[0:2]

    # last 2 points are points at the right side of the card
    arrRight = cornerPoints[2:4]

    # sort from y value for classifying top point and bottom point
    arrLeft = arrLeft[arrLeft[:, 1].argsort()]
    arrRight = arrRight[arrRight[:, 1].argsort()]

    # reorder points
    #   0--------------1
    #   |              |
    #   |              |
    #   |              |
    #   2--------------3
    reOrderCornerPoints[0] = arrLeft[0]
    reOrderCornerPoints[2] = arrLeft[1]

    reOrderCornerPoints[1] = arrRight[0]
    reOrderCornerPoints[3] = arrRight[1]

    return reOrderCornerPoints.reshape((4, 1, 2))


def getInformationFromCard(binaryCardImage):
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
    idArea = citizenCardArea['id']
    idCropImage = binaryCardImage[
        idArea['start'][1]:idArea['end'][1],
        idArea['start'][0]:idArea['end'][0]
    ]

    # custom_config = r'-l tha+eng --oem 3 --psm 6'
    idText = pytesseract.image_to_string(
        image=idCropImage,
        config=r'-c tessedit_char_whitelist=0123456789 --oem 3 --psm 6'
    )[:-2]

    # ------------------------------------------------------------------- #

    nameArea = citizenCardArea['name']
    nameCropImage = binaryCardImage[
        nameArea['start'][1]:nameArea['end'][1],
        nameArea['start'][0]:nameArea['end'][0]
    ]
    nameCropImage = cv2.dilate(nameCropImage, np.ones((3, 3)), iterations=1)
    nameText = pytesseract.image_to_string(
        image=nameCropImage,
        lang=r'tha',
        config=r'--oem 3 --psm 7'
    )[:-2]

    # ------------------------------------------------------------------- #

    dateOfBirthArea = citizenCardArea['dateOfBirth']
    dateOfBirthCropImage = binaryCardImage[
        dateOfBirthArea['start'][1]:dateOfBirthArea['end'][1],
        dateOfBirthArea['start'][0]:dateOfBirthArea['end'][0]
    ]
    dateOfBirthCropImage = cv2.dilate(dateOfBirthCropImage, np.ones((3, 3)), iterations=1)
    dateOfBirthText = pytesseract.image_to_string(
        image=dateOfBirthCropImage,
        lang=r'tha',
        config=r'--oem 3 --psm 7'
    )[:-2]

    # ------------------------------------------------------------------- #

    addressArea = citizenCardArea['address']
    addressCropImage = binaryCardImage[
        addressArea['start'][1]:addressArea['end'][1],
        addressArea['start'][0]:addressArea['end'][0]
    ]
    addressText = pytesseract.image_to_string(
        image=addressCropImage,
        lang=r'tha',
        config=r'--oem 3 --psm 6'
    )[:-2]

    return [idText, nameText, dateOfBirthText, addressText]