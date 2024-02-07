import cv2
import numpy as np


def remove_shadows(image):
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(150, 150))
   cl = clahe.apply(l)
   limg = cv2.merge((cl, a, b))
   result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
   return result


input_path = '/content/ShadowRemoval1.jpg'
output_path = '/content/ShadowRemoval1_ans.jpg'

input_image = cv2.imread(input_path)
output_image = remove_shadows(input_image)

cv2.imwrite(output_path, output_image)