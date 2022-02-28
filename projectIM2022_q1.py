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
                      width: int,
                      height: int) -> dict:
    """
    gathers all the information about the scroll found
    :param x: left top x value
    :param y: right bottom y value
    :param width: the width of the rectangle that covers the scroll
    :param height: the height of the rectangle that covers the scroll
    :return: all the information gathered
    """

    return {
        "distance": math.sqrt(x ** 2 + y ** 2),
        "left": x,
        "top": y,
        "width": width,
        "height": height
    }


def import_boxing_information_to_csv_file(found_scroll_info: list[dict],
                                          image_name: str):
    """
    imports all the information of the found scrolls to a csv file
    :param found_scroll_info: the info of each scroll in a list
    :param image_name: the name of the image
    """

    all_coordinates = []
    for data in found_scroll_info:
        points_as_in_instructions = [data['left'], data['top'], data['left'] + data['width'],
                                     data['top'] + data['height']]
        top_left_coordinate = (data['left'], data['top'])
        bottom_right_coordinate = (data['left'] + data['width'], data['top'] + data['height'])

        all_coordinates.append([points_as_in_instructions, top_left_coordinate, bottom_right_coordinate])

    df = pd.DataFrame(
        columns=['coordinates in the format of the instructions', 'top left coordinate', 'bottom right coordinate'],
        data=all_coordinates)
    df.index = np.arange(1, len(df) + 1)

    df.to_csv(f'{image_name}_boxing.csv')


def drawBoundingBoxes(img: np.ndarray,
                      scrolls_found_info: list[dict],
                      image_name: str,
                      color: tuple = (0, 0, 0)):
    """Draw bounding boxes on an image.
    :param img: image data
    :param scrolls_found_info: the data about each scroll found
    :param image_name: the name image
    :param color: Bounding box color candidates, list of RGB tuples.
    """
    for scroll in scrolls_found_info:
        left = scroll['left']
        top = scroll['top']
        right = scroll['left'] + scroll['width']
        bottom = scroll['top'] + scroll['height']
        label = scroll['label']
        image_height, image_width, _ = img.shape
        thick = int((image_height + image_width) // 900)
        cv2.rectangle(img, (left, top), (right, bottom), color, thick)
        cv2.putText(img, label, (left, top - 12), 0, 1e-3 * image_height, color, thick // 3)
    cv2.imshow(f'Boxing scrolls {image_name}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_scrolls(img: np.ndarray,
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
                scrolls_found.append(found_scroll_info(x, y, width, height))
        else:
            if small_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, width, height))
            elif medium_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, width, height))
            elif large_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, width, height))
            elif huge_scrolls(width, height):
                scrolls_found.append(found_scroll_info(x, y, width, height))

    if len(scrolls_found) < 10:
        detect_scrolls(img,
                       True,
                       image_name)
    else:
        scrolls_found.sort(key=lambda x: x['distance'])
        label_index = 1
        for scroll in scrolls_found:
            scroll['label'] = str(label_index)
            label_index = label_index + 1

        import_boxing_information_to_csv_file(found_scroll_info=scrolls_found,
                                              image_name=image_name)

        drawBoundingBoxes(img=resized_img,
                          scrolls_found_info=scrolls_found,
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

        detect_scrolls(img=resized_img,
                       add_threshold=False,
                       image_name=f'image{image_index}')
