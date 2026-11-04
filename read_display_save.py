import cv2 

# Read an image 
img = cv2.imread("photo.jpg")

if img is None:
    print("Image not found!, Check the file name and path.")
    exit()

# Inspect the image 
print("Shape", img.shape)
print("Size", img.size)
print("Type", img.dtype)

# Show the image 
cv2.imshow("Testing photo", img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite("copy-img.png", img)

# read as grayscale 
gray = cv2.imread("photo.jpg",cv2.IMREAD_GRAYSCALE)
print(gray.shape)

# save with quality 
cv2.imwrite("small.jpg", img , [cv2.IMWRITE_JPEG_QUALITY, 70])

# practice 
gray_img = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("gray-photo.png", gray_img )
show_gray_img = cv2.imread("gray-photo.png")
cv2.imshow("Show Practice Img:", show_gray_img )
cv2.waitKey(1000)
cv2.destroyAllWindows()