''' ========== Constants ========== '''
MODE = 1
resizeWidth = 800  # image width for resizing
resizeHeight = 600  # image height for resizing
kSize = 3  # size of kernel for dilation and erosion

cardWidth = 1376  # card width (8.6 x 160)
cardHeight = 864  # card height (5.4 x 160)

# location of information in the card (from 1376x864)
citizenCardArea = {
    'id': {'start': (606, 92), 'end': (1085, 154)},
    'name': {'start': (395, 156), 'end': (1350, 256)},
    'dateOfBirth': {'start': (603, 378), 'end': (900, 442)},
    'address': {'start': (137, 553), 'end': (1020, 686)}
}
