import cv2
import imutils
import numpy as np
from math import atan2, cos, sin, sqrt, pi

class DetectedObject():
    def __init__(self, img, mask, keypoints):
        self.updateObject(img, mask, keypoints, True)
    
    def updateObject(self, img, mask, keypoints, tracking=True):
        self.img = img
        self.mask = mask
        self.keypoints = keypoints
        self.tracking = tracking
    
    # Draws outline of largest white area of mask
    def drawOutline(self, color=(0,255,0), thickness=10):
        gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        cv2.drawContours(self.img, [c], -1, color, thickness)

    def getOrientation(self):
        def drawAxis(img, p_, q_, colour, scale):
            p = list(p_)
            q = list(q_)
        
            angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
            hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
            # Here we lengthen the arrow by a factor of scale
            q[0] = p[0] - scale * hypotenuse * cos(angle)
            q[1] = p[1] - scale * hypotenuse * sin(angle)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 10, cv2.LINE_AA)
            # create the arrow hooks
            p[0] = q[0] + 9 * cos(angle + pi / 4)
            p[1] = q[1] + 9 * sin(angle + pi / 4)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 10, cv2.LINE_AA)
            p[0] = q[0] + 9 * cos(angle - pi / 4)
            p[1] = q[1] + 9 * sin(angle - pi / 4)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 10, cv2.LINE_AA)
    
        def getOrientationFromPCA(pts, img):
            sz = len(pts)
            data_pts = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_pts.shape[0]):
                data_pts[i,0] = pts[i,0,0]
                data_pts[i,1] = pts[i,0,1]
            # Perform PCA analysis
            mean = np.empty((0))
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
            # Store the center of the object
            cntr = (int(mean[0,0]), int(mean[0,1]))
            
            cv2.circle(img, cntr, 3, (255, 0, 255), 2)
            p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
            p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
            drawAxis(img, cntr, p1, (0, 255, 0), 1)
            drawAxis(img, cntr, p2, (0, 0, 255), 1)
            angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
            return angle
        
        ## Apply some image preprocessing and grab contours
        gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        
        # area = cv2.contourArea(c)
        # if area < 1e2 or 1e5 < area:
        #    continue
        
        return getOrientationFromPCA(c, self.img)
        
class Micropipette(DetectedObject):
    def getTipPixel(self):
        print("Find")

# Functions
def drawPixelLocation(img, pixel, color=(255,0,0), radius=20, thickness=-1):
    cv2.circle(img, pixel, radius, color, thickness)
    
def displayImage(img):
    while True:
        cv2.imshow("Display", cv2.resize(img, (1280, 720)))
        k = cv2.waitKey(1)
        if k == 27:
            break
        

def identifyObjectInImage(img_path, mask_path):
    # Read original image and black and white mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    
    # Create Object
    micropipette = Micropipette(img, mask, [])
    micropipette.drawOutline()
    angle = micropipette.getOrientation()
    print("Angle in radians: ", angle)
    print("Angle in degrees: ", angle*(360/(2/pi)))
    
    displayImage(img)
    
if __name__=="__main__":
    identifyObjectInImage("Image.jpg", "Image_Mask.png")
    
#
#    """
#    - properties common to all objects
#        -class name
#        -keypoint list
#        -mask
#        -tracking active or not
#
#- methods common to all objects
#        -copy here, adjustable transparency, live action or not.
#        -move keypoints
#        -update transformation
#        -track
#  Each type of object can also have specialized keypoints that we insist be computed. Obvious example is the pipette tip point. Another example is well A1 of a 96-well plate.
#    """
