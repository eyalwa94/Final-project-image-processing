import copy
import math

import cv2
import numpy as np

def tiny_scrolls(width, height):
    if 10 < width < 70 and 10 < height < 70:
        return True
    return False

def small_scrolls(width, height):
    if 30 < width < 70 and 30 < height < 70:
        return True
    return False

def medium_scrolls(width, height):
    if 30 < width < 180 and 30 < height < 180:
        return True
    return False

def large_scrolls(width, height):
    if 30 < width < 350 and 30 < height < 500:
        return True
    return False

def huge_scrolls(width, height):
    if 30 < width < 600 and 30 < height < 600:
        return True
    return False

def found_scroll_info(x, y, width, height):
    return {
        "distance": math.sqrt(x ** 2 + y ** 2),
        "left": x,
        "top": y,
        "width": width,
        "height": height
    }

def drawBoundingBoxes(imageData, inferenceResults, color = (0,0,0)):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    for res in inferenceResults:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['left']) + int(res['width'])
        bottom = int(res['top']) + int(res['height'])
        label = res['label']
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        print(left, top, right, bottom)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick//3)
    cv2.imshow('Boxing scrolls', imageData)
    cv2.waitKey(0)


def seperate_view(img,
                  add):

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if add:
        h, s, v = cv2.split(hsvImg)

        # threshold saturation image
        # thresh1 = cv2.threshold(h, 70, 240, cv2.THRESH_BINARY)[1]

        # threshold value image and invert
        thresh = cv2.threshold(v, 165, 255, cv2.THRESH_BINARY)[1]
        thresh = 255 - thresh

        # combine the two threshold images as a mask
        mask = thresh

    low_white = np.array([0, 0, 206])
    high_white = np.array([255, 75, 255])
    white_mask = cv2.inRange(hsvImg, low_white, high_white)

    if add:
        hsvImg[mask != 0] = (0, 0, 0)

    hsvImg[white_mask != 0] = (0,0,0)
    new = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    contour, hierarchy = cv2.findContours(image=gray, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(len(contour)):
    #
    #     if hierarchy[0][i][3] == -1:  # external contour
    #         cv2.drawContours(image=resized, contours=contour, contourIdx=i, color=(0, 255, 0), thickness=1)

    results = []
    for c in contour:
        x, y, width, height = cv2.boundingRect(c)

        if add:
            if tiny_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height))
        else:
            if small_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height))

            elif medium_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height))

            elif large_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height))

            elif huge_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height))

    if len(results) < 10:
        seperate_view(img,
                      True)
    else:
        results.sort(key=lambda x: x['distance'])
        label_index = 1
        for result in results:
            result['label'] = str(label_index)
            label_index = label_index + 1

        drawBoundingBoxes(imageData=resized, inferenceResults=results)


if __name__ == '__main__':
    img = cv2.imread('images/image1.jpg')

    scale_percent = 15  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    check = copy.copy(resized)
    # cv2.imshow("img",img)

    seperate_view(check, False)