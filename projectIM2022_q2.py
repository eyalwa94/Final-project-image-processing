import copy
import math

import cv2
import numpy as np
import pandas as pd


# ---- scrolls possible sizes------#

def tiny_scrolls(width: int,
                 height: int) -> bool:
    if 10 < width < 70 and 10 < height < 70:
        return True
    return False


def small_scrolls(width: int,
                  height: int) -> bool:
    if 30 < width < 70 and 30 < height < 70:
        return True
    return False


def medium_scrolls(width: int,
                   height: int) -> bool:
    if 30 < width < 180 and 30 < height < 180:
        return True
    return False


def large_scrolls(width: int,
                  height: int) -> bool:
    if 30 < width < 350 and 30 < height < 500:
        return True
    return False


def huge_scrolls(width: int,
                 height: int) -> bool:
    if 30 < width < 600 and 30 < height < 600:
        return True
    return False


def found_scroll_info(x: int,
                      y: int,
                      contour: np.ndarray) -> dict:
    """
    gathers all the information about the scroll found
    :param contour: the contour coordinates
    :param x: left top x value
    :param y: right bottom y value
    :return: all the information gathered
    """

    return {
        "distance": math.sqrt(x ** 2 + y ** 2),
        "x": x,
        "y": y,
        "contour": contour
    }


def import_contour_information_to_csv_file(found_scroll_info: list[dict],
                                           image_name: str):
    """
    imports all the information of the found scrolls contour to a csv file
    :param found_scroll_info: the info of each scroll in a list
    :param image_name: the name of the image
    """

    all_coordinates = []
    for data in found_scroll_info:
        contour_to_list = list(data['contour'])
        contour_coordinates = [(element[0][0], element[0][1]) for element in contour_to_list]

        all_coordinates.append([len(data['contour']), contour_coordinates])

    df = pd.DataFrame(columns=['total number of points in contour', 'contour coordinates'],
                      data=all_coordinates)
    df.index = np.arange(1, len(df)+1)

    df.to_csv(f'{image_name}_contour.csv')


def draw_contour(scrolls_info: list[dict],
                 image_name: str):
    """
    draws the contours found
    :param scrolls_info: the info of each scroll
    :param image_name: the image name
    """
    for scroll in scrolls_info:
        text = scroll['label']
        cv2.putText(resized_img, text, (scroll['x'], scroll['y']), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 1)

        for i in range(len(scroll['contour'])):
            cv2.drawContours(image=resized_img, contours=scroll['contour'], contourIdx=i, color=(0, 255, 0), thickness=2)

    cv2.imshow(f'Contour {image_name}', resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_scrolls_contour(img: np.ndarray,
                           add_threshold: bool,
                           image_name: str) -> None:
    """
    Detects the scrolls using several functions including threshold, contour, and more
    :param img: the image
    :param add_threshold: some images need double threshold to detect the scrolls
    :param image_name: the image name
    """

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if add_threshold:
        h, s, v = cv2.split(hsvImg)

        thresh = cv2.threshold(v, 165, 255, cv2.THRESH_BINARY)[1]
        thresh = 255 - thresh

        mask = thresh

    low_white = np.array([0, 0, 206])
    high_white = np.array([255, 75, 255])
    white_mask = cv2.inRange(hsvImg, low_white, high_white)

    if add_threshold:
        hsvImg[mask != 0] = (0, 0, 0)

    hsvImg[white_mask != 0] = (0, 0, 0)
    new = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    contour, hierarchy = cv2.findContours(image=gray, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    scrolls_found = []

    for c in contour:
        x, y, width, height = cv2.boundingRect(c)

        if add_threshold:
            if tiny_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, c))
        else:
            if small_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, c))
            elif medium_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, c))
            elif large_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, c))
            elif huge_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, c))

    if len(scrolls_found) < 10:
        detect_scrolls_contour(img,
                               True,
                               image_name)
    else:
        scrolls_found.sort(key=lambda x: x['distance'])
        label_index = 1
        for scroll in scrolls_found:
            scroll['label'] = str(label_index)
            label_index = label_index + 1

        import_contour_information_to_csv_file(found_scroll_info=scrolls_found,
                                               image_name=image_name)
        draw_contour(scrolls_info=scrolls_found,
                     image_name=image_name)


if __name__ == '__main__':
    number_of_images = 8
    for image_index in range(1, number_of_images + 1):
        img = cv2.imread(f'images/image{image_index}.jpg')

        scale_percent = 15  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_img = copy.copy(resized)

        detect_scrolls_contour(img=resized_img,
                               add_threshold=False,
                               image_name=f'image{image_index}')
