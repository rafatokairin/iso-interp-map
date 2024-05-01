import cv2
import numpy as np

GRADIENT = 100

# init param
num_isoc = 10
slider = 50

def compute_contours():
    iso_contours = np.linspace(0, 1, num_isoc)  # normalized
    contours_list = []
    for isoc in iso_contours:
        # finding contours
        contours, _ = cv2.findContours(((normalized >= isoc) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours)
    return contours_list

def update_image(slider_pos):
    # interpolation
    interpolations = []
    for i in range(GRADIENT):
        if i < slider_pos:
            percent = i / slider_pos
            interp = (1 - percent) * color1 + percent * ((color1 + color2) / 2)
        else:
            percent = (i - slider_pos) / (GRADIENT - slider_pos)
            interp = (1 - percent) * ((color1 + color2) / 2) + percent * color2
        interpolations.append(interp)

    img_output = np.zeros((grayscale.shape[0], grayscale.shape[1], 3), dtype=np.uint8)
    # coloring
    for idx, isoc in enumerate(iso_contours):
        # (contour - interp) mapping
        interp_mapped = interpolations[int(isoc * (GRADIENT - 1))]
        # painting
        cv2.drawContours(img_output, contours_list[idx], -1, tuple(interp_mapped), thickness=cv2.FILLED)

    cv2.imshow('output', img_output)

grayscale = cv2.imread('images/input.jpg', cv2.IMREAD_GRAYSCALE)
normalized = grayscale.astype(np.float32) / 255.0

color1 = np.array([0, 255, 0])  # green (openCV)
color2 = np.array([0, 0, 255])  # red (openCV)

contours_list = compute_contours()
iso_contours = np.linspace(0, 1, num_isoc)  # normalized

# result
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.createTrackbar('slider', 'output', slider, GRADIENT, update_image)
update_image(slider)
cv2.waitKey(0)
cv2.destroyAllWindows()