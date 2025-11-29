##AER 850 Project 3
#EMILY PEELAR- 501169755


#importing the library for OpenCV
import cv2
#importing the numpy library
import numpy as np

#importing the image to be used through the OpenCV library
image = cv2.imread(r"C:/Projects/Project3/Project 3 Data/Project 3 Data/motherboard_image.JPEG")



#importing the library to display plots
import matplotlib.pyplot as plt

#changing the image to grayscale so that it can be used in thresholding
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#showing the image in grayscale
plt.figure()
plt.imshow(gray_image, cmap='gray')
plt.title("Image in Grayscale")
plt.axis('off')
plt.show()



#creating the binary threshold with blur
_,threshold = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)

#showing the image with the binary threshold
plt.figure()
plt.imshow(threshold, cmap='gray')
plt.title("Image with Binary Threshold")
plt.axis('off')
plt.show()


#using canny to detect edges
edges = cv2.Canny(threshold, 100, 200)
plt.figure()
plt.imshow(edges, cmap='gray')
plt.title("Image with Edges Defined with Canny")
plt.axis('off')
plt.show()





#trying with a blur
#adding gaussian blur to help differentiate between key features:
blur = cv2.GaussianBlur(gray_image, (5, 5),0)

#creating the binary threshold with blur
_,threshold_blurred = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

#showing the image with the binary threshold
plt.figure()
plt.imshow(threshold_blurred, cmap='gray')
plt.title("Image with Binary Threshold and blurred")
plt.axis('off')
plt.show()

#using canny to detect edges
edges = cv2.Canny(threshold_blurred, 100, 200)
plt.figure()
plt.imshow(edges, cmap='gray')
plt.title("Image with Edges Defined with Canny and blurred")
plt.axis('off')
plt.show()







#now using the harris method for corner detection
#harris requires a 32float image in grayscale
float_32_gray = np.float32(gray_image)
#adding blocksize, aperture size, and k-values
dst_harris = cv2.cornerHarris(float_32_gray,2,3,0.04)
#dilation to make corners more visible
dilation_harris= cv2.dilate(dst_harris, None)
#the image is copied so that it can be modified with indicators for the corners
harris_copied = image.copy()
#marking the corners with red dots with 1% of the maximum values
harris_copied[dilation_harris > 0.01 * dilation_harris.max()] = [0, 0, 255]

#plotting the harris method and corners detected
plt.figure()
#the image colour channels were swapped for plotting
plt.imshow(cv2.cvtColor(harris_copied, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection with a Maximum value of 1%")
plt.axis("off")
plt.show()

#now attempting a harris with a maximum value of 0.05
#marking the corners with red dots with 5% of the maximum values
#the image is copied so that it can be modified with indicators for the corners
harris_copied_five = image.copy()
harris_copied_five[dilation_harris > 0.05 * dilation_harris.max()] = [0, 0, 255]

#plotting the harris method and corners detected
plt.figure()
#the image colour channels were swapped for plotting
plt.imshow(cv2.cvtColor(harris_copied_five, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection with a Maximum value of 5%")
plt.axis("off")
plt.show()

