import numpy as np
import argparse
import cv2
import imutils 


def seg_img(img, fn):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0+mask1

    output_img = img.copy()
    output_img[np.where(mask==0)] = 0
    cv2.imwrite("images-out-"+fn, output_img)

    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    _, clone = cv2.threshold(gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # find contours in the thresholded image, then initialize the
    # list of group locations
    groupCnts = cv2.findContours(blur, 
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
    groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
    groupLocs = []

    def union(a,b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def intersection(a,b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0: 
            return ()
        return (x, y, w, h)
    # loop over the group contours
    for (i, c) in enumerate(groupCnts):
        # compute the bounding box of the contour    
        a = cv2.boundingRect(c)
        # only accept the contour region as a grouping of characters if
        # the ROI is sufficiently large
        if a[2] < 50 or a[3] < 50:
            continue
        for (j, d) in enumerate(groupCnts):
        # compute the bounding box of the contour
            if j == i:
                continue
            b = cv2.boundingRect(d)
            inter = intersection(a, b)
            if len(inter) > 0:
                a = union(a, b)
        groupLocs.append(a)
    groupRec = []
    for i in range(len(groupLocs)):
        found = False
        for j in range(len(groupRec)):
            if groupLocs[i][0] == groupRec[j][0] and groupLocs[i][1] == groupRec[j][1] and groupLocs[i][2]== groupRec[j][2] and groupLocs[i][3] == groupRec[j][3]:
                found = True
                break
        if not found:
            groupRec.append(groupLocs[i])

    return clone, groupRec
def detect_rec(img, fn):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    _, threshed = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)

    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    canvas  = img.copy()
    cnts = sorted(cnts, key = cv2.contourArea)
    cnt = cnts[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

def test_it():
    parser = argparse.ArgumentParser(description='color segmentation')
    parser.add_argument('--input', help='Path to input image.', default='image.jpg')
    args = parser.parse_args()
    img = cv2.imread(args.input)
    if img is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    srcClone = img.copy()
    fn = args.input
    image, locs = seg_img(img, fn)
    c = 0
    for i in locs:
        im = image[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]   
        cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
        cv2.rectangle(srcClone, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0,0,0), 10)
        cv2.rectangle(image, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0,0,0), 10)
        c = c + 1
    print(locs)
    x, y, w, h = detect_rec(img, fn)
    cv2.rectangle(srcClone, (x, y), (x + w, y + h), (255, 255, 255), 10)
    cv2.imwrite("seg-all-" + fn, srcClone)
    cv2.imwrite("seg-img-" + fn, image)

if __name__ == '__main__':
    test_it()
