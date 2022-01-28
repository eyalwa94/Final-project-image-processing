import cv2
import numpy as np
import matplotlib.pyplot as plt



def main():
    """
    not copied from https://github.com/ierolsen/Object-Detection-with-OpenCV
    :return:
    """

    img = cv2.imread('images/image8.jpg')

    scale_percent = 15 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    coin_blur = cv2.medianBlur(src=resized, ksize=3)




    coin_gray = cv2.cvtColor(coin_blur, cv2.COLOR_BGR2GRAY)
    coin_gray2 = cv2.bitwise_not(coin_gray)

    ret, coin_thres = cv2.threshold(src=coin_gray2, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

    contour, hierarchy = cv2.findContours(image=coin_thres.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contour)):

        if hierarchy[0][i][3] == -1:  # external contour
            cv2.drawContours(image=resized, contours=contour, contourIdx=i, color=(0, 255, 0), thickness=3)

    cv2.imshow('img', resized)
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()