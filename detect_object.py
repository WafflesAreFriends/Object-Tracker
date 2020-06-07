import cv2
import imutils

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
        
class Micropipette(DetectedObject):
    def getTipPixel():
        print("Find")


# Functions
def drawPixelLocation(img, pixel, color=(255,0,0), radius=20, thickness=-1):
    cv2.circle(img, pixel, radius, color, thickness)

def identifyObjectInImage(img_path, mask_path):
    # Read original image and black and white mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    
    # Create Object
    micropipette = Micropipette(img, mask, [])
    micropipette.drawOutline()
    
    drawPixelLocation(img, (50, 50))
    
    cv2.imshow("Show", cv2.resize(img, (1280, 720)))
    cv2.waitKey(2000)
    
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
