#  import an image and show it

import cv2
import numpy as np


# read the image
img = cv2.imread('image.jpeg')


# show this image in grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#  show this image in 1 bit color
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


# Apply random noise to the grayscale image
noise = np.zeros(gray.shape, np.uint8)
cv2.randn(noise, 0, 150)
noisy = cv2.add(gray, noise)


# apply the floyd-steinberg dithering algorithm to the grayscale image
def floyd_steinberg_dithering(image):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_pixel = image[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x + 1 < image.shape[1]:
                image[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < image.shape[0]:
                image[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < image.shape[0]:
                image[y + 1, x] += quant_error * 5 / 16
            if x + 1 < image.shape[1] and y + 1 < image.shape[0]:
                image[y + 1, x + 1] += quant_error * 1 / 16
    return image


floyd = floyd_steinberg_dithering(gray.copy())


# Apply dithering bayer matrix to the grayscale image
def bayer_dithering_4x4(image):
    bayer = np.array([[0, 8, 2, 10],
                      [12, 4, 14, 6],
                      [3, 11, 1, 9],
                      [15, 7, 13, 5]])
    bayer = bayer * 16
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_pixel = image[y, x]
            new_pixel = 255 if old_pixel > bayer[y % 4, x % 4] else 0
            image[y, x] = new_pixel
    return image

# Apply dithering Ordered matrix of size 2x2 to the grayscale image


def bayer_dithering_2x2(image):
    ordered = np.array([[0, 2],
                        [3, 1]])
    ordered = ordered * 85
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            old_pixel = image[y, x]
            new_pixel = 255 if old_pixel > ordered[y % 2, x % 2] else 0
            image[y, x] = new_pixel
    return image


#  show the image
bayer4 = bayer_dithering_4x4(gray.copy())
bayer2 = bayer_dithering_2x2(gray.copy())

# show all these images in the same window using subplot in full screen in 3 rows and 3 2 columns
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow('image', np.hstack((np.vstack((gray, floyd)), np.vstack(
    (thresh, bayer4)), np.vstack((noisy, bayer2)))))

cv2.waitKey(0)
cv2.destroyAllWindows()
