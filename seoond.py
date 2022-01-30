import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    not copied from https://github.com/ierolsen/Object-Detection-with-OpenCV
    :return:
    """

    img = cv2.imread('images/image1.jpg')

    scale_percent = 15 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    check = copy.copy(resized)
    coin_blur = cv2.medianBlur(src=resized, ksize=3)

    coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
    coin_gray2 = cv2.bitwise_not(coin_gray)

    ret, coin_thres = cv2.threshold(src=coin_gray2, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

    # Works good show contour
    contour, hierarchy = cv2.findContours(image=coin_thres.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contour)):

        if hierarchy[0][i][3] == -1:  # external contour
            print(f'conutre index {i}')
            cv2.drawContours(image=resized, contours=contour, contourIdx=i, color=(0, 255, 0), thickness=3)

    idx = 0
    for c in contour:
        x, y, w, h = cv2.boundingRect(c)

        if  10 < w < 400 and 10 < h < 400:
            cv2.rectangle(resized, (x, y + h), (x + w, y), (100, 100, 100), 3)
            idx += 1
            new_img = resized[y:y + h, x:x + w]
    # kernel = np.ones((3, 3), np.uint8)
    #
    # opening = cv2.morphologyEx(coin_thres, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    #
    # dist_transform = cv2.distanceTransform(src=opening, distanceType=cv2.DIST_L2, maskSize=5)
    #
    # ret, sure_foreground = cv2.threshold(src=dist_transform, thresh=0.4 * np.max(dist_transform), maxval=255, type=0)
    #
    # sure_background = cv2.dilate(src=opening, kernel=kernel, iterations=1)  # int
    #
    # sure_foreground = np.uint8(sure_foreground)  # change its format to int
    #
    # unknown = cv2.subtract(sure_background, sure_foreground)
    #
    # ret, marker = cv2.connectedComponents(sure_foreground)
    #
    # marker = marker + 1
    #
    # marker[unknown == 255] = 0  # White area is turned into Black to find island for watershed
    #
    # marker = cv2.watershed(image=resized, markers=marker)
    #
    # marker = marker.astype(np.uint8)
    #
    # contour, hierarchy = cv2.findContours(image=marker.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(len(contour)):
    #
    #     if hierarchy[0][i][3] == -1:
    #         cv2.drawContours(image=resized, contours=contour, contourIdx=i, color=(255, 0, 0), thickness=3)

    cv2.imshow('img', resized)
    cv2.imshow('img2', check)
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()