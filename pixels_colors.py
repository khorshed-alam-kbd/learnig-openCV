import cv2

img = cv2.imread("photo.jpg")

# Color of the pixel at x=100, y=50
b, g, r = img[50, 100]
print("Blue:", b, "Green:", g, "Red:", r)

# Make one pixel red (BGR order)
img[50, 100] = [0, 0, 255]

# Paint a whole region green (y: 0–100, x: 0–200)
img[0:200, 0:200] = [0, 255, 0]

cv2.imshow("Edited", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
