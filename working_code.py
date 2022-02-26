import copy
import math

import cv2
import numpy as np
import pandas as pd


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


def found_scroll_info(x, y, width, height, c):
    return {
        "distance": math.sqrt(x ** 2 + y ** 2),
        "left": x,
        "top": y,
        "width": width,
        "height": height,
        "c": c,
        "x": x,
        "y": y
    }


def import_boxing_information_to_csv_file(found_scroll_info,
                                          image_name):

    all_coordinates = []
    for data in found_scroll_info:
        points_as_in_instructions = [data['left'], data['top'], data['left'] + data['width'], data['top'] + data['height']]
        top_left_coordinate = (data['left'], data['top'])
        bottom_right_coordinate = (data['left'] + data['width'], data['top'] + data['height'])

        all_coordinates.append([points_as_in_instructions, top_left_coordinate, bottom_right_coordinate])

    df = pd.DataFrame(columns=['coordinates in the format of the instructions', 'top left coordinate', 'bottom right coordinate'],
                      data=all_coordinates)
    df.index = np.arange(1, len(df) + 1)

    df.to_csv(f'{image_name}.csv')


def import_contour_information_to_csv_file(found_scroll_info,
                                           image_name):

    all_coordinates = []
    single = []
    singles = []
    for data in found_scroll_info:
        # all_coordinates.append([tuple(data['c'])])
        c_to_list = list(data['c'])
        check = [(element[0][0], element[0][1]) for element in c_to_list]

        all_coordinates.append([len(data['c']), check])

    df = pd.DataFrame(columns=['total number of points in contour', 'contour coordinates'],
                      data=all_coordinates)
    df.index = np.arange(1, len(df)+1)

    df.to_csv(f'{image_name}_check.csv')


def drawBoundingBoxes(imageData, inferenceResults, image_name, contour_image, color=(0, 0, 0), ):
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
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick//3)
    cv2.imshow(f'Boxing scrolls {image_name}', imageData)
    cv2.imshow(f'Conture {image_name}', contour_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def do_it(c, x, y, index):
    text = "original, num_pts={}".format(len(c))
    cv2.putText(resized_img_contour_2, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 1)

    # show the original contour image
    for i in range(len(c)):
        cv2.drawContours(image=resized_img_contour_2, contours=c, contourIdx=i, color=(0, 255, 0), thickness=2)


def do_it_2(results):
    for result in results:
        text = result['label']
        cv2.putText(resized_img_contour_2, text, (result['x'], result['y']), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 1)

        # show the original contour image
        print("[INFO] {}".format(text))
        for i in range(len(result['c'])):
            cv2.drawContours(image=resized_img_contour_2, contours=result['c'], contourIdx=i, color=(0, 255, 0), thickness=2)


def seperate_view(img,
                  add,
                  image_name):

    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if add:
        h, s, v = cv2.split(hsvImg)

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

    hsvImg[white_mask != 0] = (0, 0, 0)
    new = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

    contour, hierarchy = cv2.findContours(image=gray, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contour)):

        if hierarchy[0][i][3] == -1:  # external contour
            cv2.drawContours(image=resized_img_contour, contours=contour, contourIdx=i, color=(0, 255, 0), thickness=2)

    results = []
    for c in contour:
        x, y, width, height = cv2.boundingRect(c)

        if add:
            if tiny_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height, c))
                # do_it(c, x, y)
        else:
            if small_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height, c))
                # do_it(c, x, y)
                # print(f'contoure is {c}')
            elif medium_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height, c))
                # do_it(c, x, y)
                # print(f'contoure is {c}')
            elif large_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height, c))
                # do_it(c, x, y)
                # print(f'contoure is {c}')
            elif huge_scrolls(width, height):
                results.append(found_scroll_info(x, y, width, height, c))
                # do_it(c, x, y)
                # print(f'contoure is {c}')

    if len(results) < 10:
        seperate_view(img,
                      True,
                      image_name)
    else:
        results.sort(key=lambda x: x['distance'])
        label_index = 1
        for result in results:
            result['label'] = str(label_index)
            label_index = label_index + 1

        do_it_2(results)
        import_boxing_information_to_csv_file(results, image_name)
        import_contour_information_to_csv_file(results, image_name)
        drawBoundingBoxes(imageData=resized, inferenceResults=results, image_name=image_name, contour_image=resized_img_contour_2)


if __name__ == '__main__':
    number_of_images = 8
    for image_index in range(1, number_of_images+1):
        img = cv2.imread(f'images/image{image_index}.jpg')

        scale_percent = 15  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_img = copy.copy(resized)
        resized_img_contour = copy.copy(resized)
        resized_img_contour_2 = copy.copy(resized)

        seperate_view(resized_img, False, f'image{image_index}')