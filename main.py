import cv2
import numpy as np

img = cv2.imread("red.png")
# resize so it fits on screen
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
# copy to add lines to
org = img
# blur makes cones stand out
img = cv2.blur(img, (10, 10))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# mask to get the cones separated
lower_cone = np.array([0, 70, 150]).astype("uint8")
upper_cone = np.array([5, 255, 250]).astype("uint8")
mask = cv2.inRange(hsv, lower_cone, upper_cone)
detected_output = cv2.bitwise_and(img, img, mask=mask)


# cut in half
height, width, channels = img.shape
left_image = mask[0:height, 0:int(width/2)]
right_image = mask[0:height, int(width/2):width]

# Extends lines to end of image
def extend_line(p1, p2, distance):
    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    p3_x = int(p1[0] + distance*np.cos(diff))
    p3_y = int(p1[1] + distance*np.sin(diff))
    p4_x = int(p1[0] - distance*np.cos(diff))
    p4_y = int(p1[1] - distance*np.sin(diff))
    return (p3_x, p3_y), (p4_x, p4_y)

# Find cones and coordinates for the end cones and draw the line between them
#    Left Cones

edgedL = cv2.Canny(left_image, 30, 200)
contoursL, hierarchyL = cv2.findContours(edgedL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# finds edge cones
size = len(contoursL)
x, y, w, h = cv2.boundingRect(contoursL[0])
x1, y1, w1, h1 = cv2.boundingRect(contoursL[size-1])
# draw line (Center line in middle of cone)
x = int(x+w/2)
y = int(y+h/2)
p1, p2 = extend_line((x, y), (x1, y1), 1000)
cv2.line(org, p1, p2, (0, 0, 255), 3)

#    Right Cones

edgedR = cv2.Canny(right_image, 30, 200)
contoursR, hierarchyR = cv2.findContours(edgedR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# finds end cones
size2 = len(contoursR)
x2, y2, w2, h2 = cv2.boundingRect(contoursR[0])
x3, y3, w3, h3 = cv2.boundingRect(contoursR[size2-2])
cv2.rectangle(right_image, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 3)

# draw line (Center line in middle of cone and add half width so its correct)
x2 = int(x2+w2/2 + width/2)
y2 = int(y2+h2/2)
x3 = int(x3+w3/2 + width/2)
y3 = int(y3+h3/2)
p3, p4 = extend_line((x2, y2), (x3, y3), 1000)
cv2.line(org, p3, p4, (0, 0, 255), 3)

# Export Image
cv2.imwrite("answer.png", org)




