import cv2 
from random import randrange
#Step by step

# 1. Our Image or Video file / Importing The Image or Video
img_file = "Humans.png"


# 2. Our pre-trained car classifier
classifier_file = "haarcscade_frontalface_default.xml" 
trained_face_data = cv2.CascadeClassifier(classifier_file)

# 3. Create opencv image / reads pixels in image file
img = cv2.imread(img_file)


# 4. covert to greyscale (needed for haar cascade)
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#6. detect cars of any size or scale/ look at open cv documentation
face_coordinates = trained_face_data.detectMultiScale(grayscale)

#7. Draw rectangles around the cars / Interpolation
for (x, y, w, h) in face_coordinates:
    # the parameters are (x, y) - top left point (x + width, y + height) - bottom right point (RGB), (width of rectangle)
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128,266), randrange(128,266), randrange(128,266)), 10)

# 8. Display the image with the faces spotted / displays image for a milisecond - need waitKey to pause the image
cv2.imshow("Clever Programmer Car Detector", img) #black_n_white to img

# 9. Don't autoclose (wait here in the cod and listen for a key press)
cv2.waitKey()

print("Code Completed")
















