import cv2
import numpy as np

def main():
    img = cv2.imread('images/image1.jpg', 0)

    scale_percent = 10  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)




    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    ####

    ### bitwise
    imagem = cv2.bitwise_not(resized)

    #good
    ret, coin_thres = cv2.threshold(src=imagem, thresh=75, maxval=255, type=cv2.THRESH_BINARY)

    #edges
    edges = cv2.Canny(imagem, 0, 200)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel=kernel, iterations=2)

    sure_background = cv2.dilate(src=opening, kernel=kernel, iterations=5)  # int

    sure_foreground = np.uint8(sure_background)  # change its format to int
    cv2.subtract(sure_background, sure_foreground)
    
    ret, coin_thres = cv2.threshold(src=sure_foreground, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

    
    cv2.imshow('img', coin_thres)
    cv2.imshow('original', resized)
    # cv2.imshow('img2', imagem)
    cv2.waitKey()
    cv2.destroyAllWindows()














if __name__ == '__main__':
    main()