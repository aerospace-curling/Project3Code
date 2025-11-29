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



#creating the binary threshold without blur
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





#after analyzing the methods above, it was decided that the blurred canny method was the optimal choice. This is because canny was able to determine continuous object edges, whereas harris only found the corner points 

#first it was attempted to use the threshold, which resulted in the findings shown
#now starting with contour detection, the blurred threshold is used again
#findCountours is used with the blurred threshold defined previously, RETR_EXTERNAL to find the external contours, and CHAIN_APPROX_SIMPLE to store all contour points 
contours, hierarchy = cv2.findContours(threshold_blurred,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#then another copy of the image is made
contour_image=image.copy()
#drawContours is added to draw contours on the original image, coloured in green and a thickness of 2
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

#now plotting the figure to see the results
plt.figure()
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection")
plt.axis("off")
plt.show()














#an inverted threshold was added being that the original threshold was just picking up the edge of the desk
_, threshold_inverted = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
#plotting with the inverted threshold
plt.figure()
plt.imshow(threshold_inverted, cmap='gray')
plt.title("Inverted Binary Threshold")
plt.axis('off')
plt.show()

#findCountours is used with the inverted and blurred threshold, RETR_EXTERNAL to find the external contours, and CHAIN_APPROX_SIMPLE to store all contour points 
contours, hierarchy = cv2.findContours(threshold_inverted,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#a minimum area was added to filter out small contours
minimum_area=50000
#an empty list was created to store the largest contours from the for loop
largest_contours=[]
#a for loop was created to loop through the contours to determine the largest one
for contour in contours:
    area = cv2.contourArea(contour)
    if area > minimum_area:
        largest_contours.append(contour)
#checking if the list is empty.If it is, it is replaced with the original contours
if len(largest_contours) == 0:
    largest_contours = contours
#finding the overall largest contour as obtained from the for loop, with the key selecting the contour with the greatest area
overall_largest_contour = max(largest_contours, key=cv2.contourArea)

#then another copy of the image is made
contour_image=image.copy()
#drawContours is added to draw the largest contour on the original image, coloured in green and a thickness of 2
cv2.drawContours(contour_image, [overall_largest_contour], -1, (0, 255, 0), 2)

#now plotting the figure to see the results
plt.figure()
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection with Inverted Threshold")
plt.axis("off")
plt.show()

#the findings of the inverted method with area defined were far superior to those using just the original threshold

        



#starting the masking steps
#first the mask size was defined to be the same as the grayscale image through the use of np.zeros_like
mask = np.zeros_like(gray_image)
#drawing the contour and filling it with white completely
cv2.drawContours(mask, [overall_largest_contour], -1, 255, thickness=cv2.FILLED)
#this line uses the mask to keep only the PCB pixels from the original image and turn everything else black
extracted_pcb = cv2.bitwise_and(image, image, mask=mask)

#creating a plot to show the results
plt.figure()
plt.imshow(cv2.cvtColor(extracted_pcb, cv2.COLOR_BGR2RGB))
plt.title("Extracted PCB")
plt.axis("off")
plt.show()

#given that the image for comparison was rotated, the extracted PCB can be rotated
rotated_pcb = cv2.rotate(extracted_pcb, cv2.ROTATE_90_CLOCKWISE)

#creating a plot to show the rotated version
plt.figure()
plt.imshow(cv2.cvtColor(rotated_pcb, cv2.COLOR_BGR2RGB))
plt.title("Extracted PCB (Rotated)")
plt.axis("off")
plt.show()




