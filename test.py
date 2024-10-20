import cv2
import numpy as np

image = cv2.imread("flask_app/static/images/scale_test2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

longest_length = 0
longest_edge = None

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)

        for i in range(4):
            point1 = approx[i][0]
            point2 = approx[(i + 1) % 4][0]
            length = np.linalg.norm(point1 - point2)

            if length > longest_length:
                longest_length = length
                longest_edge = (point1, point2)

if longest_edge is not None:
    cv2.line(image, tuple(longest_edge[0]), tuple(longest_edge[1]), (255, 0, 0), 2)

cv2.imshow("Detected Squares", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Longest Edge Length:", longest_length)
