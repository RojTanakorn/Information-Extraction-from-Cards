''' ========== Constants ========== '''
resizeWidth = 800  # image width for resizing
resizeHeight = 600  # image height for resizing
kSize = 3  # size of kernel for dilation and erosion

cardWidth = 1376  # card width (8.6 x 160)
cardHeight = 864  # card height (5.4 x 160)

CARDTYPE = ['CITIZEN', 'DRIVING']

# location of information in the card (from 1376x864)
citizenCardArea = {
    'id': {'start': (606, 92), 'end': (1085, 154)},
    'name': {'start': (395, 156), 'end': (1350, 256)},
    'dateOfBirth': {'start': (603, 378), 'end': (900, 442)},
    'address': {'start': (137, 553), 'end': (950, 686)}
}

drivingCardArea = {
    'name': {'start': (462, 370), 'end': (1120, 452)},
    'nameEng': {'start': (476, 454), 'end': (1120, 510)},
    'dateOfBirth': {'start': (537, 574), 'end': (809, 621)},
    'id': {'start': (812, 682), 'end': (1165, 727)},
}