import cv2
import numpy as np

def ShadowMask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 0], dtype=np.uint8)
    upper_bound = np.array([200, 60,100], dtype=np.uint8)
    shadow_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return shadow_mask

def interested_area(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def update_pixel(i, j, result, RemovedShadow, image, RoadSM):
    if result[i][j] == 0 or RoadSM[i][j] == 0:
        return image[i][j]
    
    neighbours = [(ni, nj) for ni, nj in [(i-1, j), (i, j-1), (i, j+1), (i+1, j)] if 0 <= ni < RemovedShadow.shape[0] and 0 <= nj < RemovedShadow.shape[1] and result[ni][nj] == 0]

    if neighbours:
        neighbours_sum = sum(RemovedShadow[ni][nj] for ni, nj in neighbours)
        return neighbours_sum / (255 * len(neighbours)) * np.array([151, 153, 143])
    
    return np.array([161, 173, 183])

image = cv2.imread("/home/shourya/Downloads/ShadowRemoval2.jpg")
image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
result = ShadowMask(image)
RemovedShadow = np.zeros_like(image)


exact_part = [(160, 415),(800, 450),(630, 190),(420, 170)]
grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
RoadSM = interested_area(grey_img, np.array([exact_part], np.int32))
RemovedShadow = np.array([[update_pixel(i, j, result, RemovedShadow, image, RoadSM) for j in range(RemovedShadow.shape[1])] for i in range(RemovedShadow.shape[0])])
cv2.imwrite("Result2.jpg", RemovedShadow)


