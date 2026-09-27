import cv2
import numpy as np

img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


noise = np.random.normal(0, 25, gray.shape)
img_gauss = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)   

img_sp = gray.copy()
p= 0.02
salt = np.random.rand(*gray.shape) < p
pepper = np.random.rand(*gray.shape) < p
img_sp[salt] = 255
img_sp[pepper] = 0

g = cv2.GaussianBlur(img_gauss, (5,5),0)
m = cv2.medianBlur(img_sp, 5)
b = cv2.bilateralFilter(img_gauss, 9, 75, 75)

cv2.imshow("gauss noisy", img_gauss)
cv2.imshow("gaussian blur", g)
cv2.imshow("salt-pepper noisy", img_sp)
cv2.imshow("median blur", m)
cv2.imshow("bilateral", b)
cv2.waitKey(0)
cv2.destroyAllWindows()

