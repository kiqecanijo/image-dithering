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
cv2.randn(noise, 0, 50)
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


grayTemp = gray.copy()
floyd = floyd_steinberg_dithering(grayTemp)

# show all these images in the same window using subplot
cv2.imshow('image', np.hstack((cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(
    thresh, cv2.COLOR_GRAY2BGR), cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR), cv2.cvtColor(floyd, cv2.COLOR_GRAY2BGR)
)
))
cv2.waitKey(0)
cv2.destroyAllWindows()
