import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)

cv2.imshow("original", img)
cv2.imshow("gray", gray)
cv2.imshow("equalized", eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.plot(hist)
plt.title("histogram")
plt.show()
