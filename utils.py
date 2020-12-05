# Import libraries
import cv2  # provides image processing tools
import numpy as np  # deal with image array
import matplotlib.pyplot as plt  # plot graph and show images
import pytesseract # provides OCR
from collections import Counter

from constants import citizenCardArea, drivingCardArea, cardWidth, cardHeight, CARDTYPE


def showImages(images, titles):
    for i in range(len(images)):
        plt.figure(titles[i])
        if images[i].ndim == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')


def preprocess(originalImage, imageWidth, imageHeight, kSize):
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
    noneCount = 0
    previousThresholds = []

    # apply Hough Transform algorithm for getting lines
    # use while loop for adjusting threshold that we can get 4 lines
    while True:
        lines = cv2.HoughLines(
            blankImage, 1, np.pi / 180, lineThreshold, None, 0, 0
        )

        # @@@ Check that HoughLines cannot find the line (get None)
        # if cannot find the line more than 3 times, return None
        if lines is None:
            if noneCount > 3:
                break
            noneCount = noneCount + 1
        # if can find the line, check num line that is equal to 4 ?
        else:
            if len(lines) == 4:
                break
            elif len(lines) > 4:
                lineThreshold = lineThreshold+1
            else:
                lineThreshold = lineThreshold-1
            
            # prevent thereshold is stuck (ex. 199 - 200 - 199 - 200 - ... - toggle)
            # if it's stuck more than 4 times, return None for capturing again
            if len(previousThresholds) == 8:
                items = Counter(previousThresholds).keys()
                if len(items) == 2:
                    lines = None
                    break
                previousThresholds = previousThresholds[1:]
            previousThresholds.append(lineThreshold)
            
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


def classifyCard(baseCardImage):
    # preprocess card image by converting to grayscale and apply filter
    grayCardImage = cv2.cvtColor(baseCardImage, cv2.COLOR_RGB2GRAY)
    grayCardImage = cv2.GaussianBlur(grayCardImage, (3, 3), 0)

    # read 2 logos as grayscale for classifier
    citizen_logo = cv2.imread('./card_logos/citizen_logo.jpg', cv2.IMREAD_GRAYSCALE)
    driving_logo = cv2.imread('./card_logos/driving_logo.jpg', cv2.IMREAD_GRAYSCALE)

    logo_list = [citizen_logo, driving_logo]
    showImages(logo_list, ['Citizen logo', 'Driving license logo'])

    # list for storing max values from matching
    maxVals = []

    templateMatchingImage = np.copy(baseCardImage)

    # using for-loop to compute matching values from each logo
    for index in range(len(logo_list)):
        # blur and apply Canny to logo
        logo = cv2.GaussianBlur(logo_list[index], (5, 5), 0)
        logo = cv2.Canny(logo, 75, 150)

        showImages([logo], [f'logo {index}'])

        # get height and width of logo
        logoHeight, logoWidth = logo.shape

        # tuple for store max value, location of max value, and rescale ratio
        found = None

        # apply multi-scale template matching
        for scale in np.linspace(0.2, 1.0, 40):
            card = cv2.resize(grayCardImage, (int(cardWidth * scale), int(cardHeight * scale)))
            rescale = grayCardImage.shape[1] / float(card.shape[1])

            if card.shape[0] < logoHeight or card.shape[1] < logoWidth:
                break

            cardWithCanny = cv2.Canny(card, 25, 150)
            result = cv2.matchTemplate(cardWithCanny, logo, cv2.TM_CCOEFF)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, rescale)

        (maxVal, maxLoc, rescale) = found
        maxVals.append(maxVal)


        (startX, startY) = (int(maxLoc[0] * rescale), int(maxLoc[1] * rescale))
        (endX, endY) = (int((maxLoc[0] + logoWidth) * rescale), int((maxLoc[1] + logoHeight) * rescale))

        if index == 0:
            color = (255, 0, 0)
            h = 290
            typ = 'citizen'
        else:
            color = (0, 255, 0)
            h = 330
            typ = 'driving'
        cv2.rectangle(templateMatchingImage, (startX, startY), (endX, endY), color, 3)
        cv2.putText(templateMatchingImage, f'{typ}: {maxVal}', (940, h), cv2.FONT_HERSHEY_PLAIN, 2.2, color, 3)

    return CARDTYPE[np.argmax(maxVals)], grayCardImage, templateMatchingImage


def getInformationFromCard(baseCardImage, binaryCardImage, cardType):
    pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
    headerList = ['เลขประจำตัวประชาชน', 'ชื่อ-สกุล', 'เกิดวันที่']

    if cardType == 'CITIZEN':
        typeArea = citizenCardArea
    elif cardType == 'DRIVING':
        typeArea = drivingCardArea

    idArea = typeArea['id']
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

    nameArea = typeArea['name']
    nameCropImage = binaryCardImage[
        nameArea['start'][1]:nameArea['end'][1],
        nameArea['start'][0]:nameArea['end'][0]
    ]
    nameText = pytesseract.image_to_string(
        image=nameCropImage,
        lang=r'tha',
        config=r'--oem 3 --psm 7'
    )[:-2]

    # ------------------------------------------------------------------- #

    dateOfBirthArea = typeArea['dateOfBirth']
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


    # ================= #
    infoList = [str(idText), nameText, dateOfBirthText]

    # ------------------------------------------------------------------- #
    # ----------------------------  ADDITION ---------------------------- #
    # ------------------------------------------------------------------- #

    if cardType == 'CITIZEN':
        # ======= Address ======= #
        addressArea = typeArea['address']
        addressCropImage = binaryCardImage[
            addressArea['start'][1]:addressArea['end'][1],
            addressArea['start'][0]:addressArea['end'][0]
        ]
        addressText = pytesseract.image_to_string(
            image=addressCropImage,
            lang=r'tha',
            config=r'--oem 3 --psm 6'
        )[:-2]

        addressText = addressText.replace('ที่อยู่ ', '')
        addressText = addressText.replace('\n', ' ')

        infoList.append(addressText)
        headerList.append('ที่อยู่')


    elif cardType == 'DRIVING':
        # ======= Name in English ======= #
        nameEngArea = typeArea['nameEng']
        nameEngCropImage = binaryCardImage[
            nameEngArea['start'][1]:nameEngArea['end'][1],
            nameEngArea['start'][0]:nameEngArea['end'][0]
        ]
        nameEngText = pytesseract.image_to_string(
            image=nameEngCropImage,
            lang=r'eng',
            config=r'--oem 3 --psm 6'
        )[:-2]

        infoList.append(nameEngText)
        headerList.append('ชื่อ-สกุล ภาษาอังกฤษ')

    infoAreaImage = np.copy(baseCardImage)
    for key in typeArea:
        keyType = typeArea[key]
        cv2.rectangle(infoAreaImage, keyType['start'], keyType['end'], (255,0,0), 3)

    return infoList, headerList, infoAreaImage