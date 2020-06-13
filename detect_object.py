import cv2
import imutils
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from scipy import ndimage as nd

##### Classes #####

class DetectedObject():
    def __init__(self, img, mask, keypoints):
        self.h, self.w, _ = img.shape
        self.xy_grid = np.meshgrid(np.arange(self.w), np.arange(self.h))

        self.updateObject(img, mask, keypoints, True)
    
    def updateObject(self, img, mask, keypoints, tracking=True):
        self.img = img
        self.mask = mask
        self.keypoints = keypoints
        self.tracking = tracking
    
    # Draws outline of largest white area of mask
    def drawOutline(self, color=(0,255,0), thickness=10):
        c = self.getContours()
        cv2.drawContours(self.img, [c], -1, color, thickness)

    # Temporary
    def getContours(self):
        gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return max(cnts, key=cv2.contourArea)

    def getPCAResults(self):
        pts = self.getContours()
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
            
        # Perform PCA analysis
        mean = np.empty((0))
        return cv2.PCACompute2(data_pts, mean) # mean, eigenvectors, eigenvalues
        #cntr = (int(mean[0,0]), int(mean[0,1])) # Store center of axes in tuple

    def getOrientation(self):
        mean, eigenvectors, eigenvalues = self.getPCAResults()
        cntr = (int(mean[0,0]), int(mean[0,1])) # Store center of axes in tuple

        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        
        
        cv2.line(self.img, (int(cntr[0]), int(cntr[1])), (int(p1[0]), int(p1[1])), (255, 0, 0), 5, cv2.LINE_AA)
        cv2.line(self.img, (int(cntr[0]), int(cntr[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 5, cv2.LINE_AA)

        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        
        pax = self.principal_axes()
        
        # Bill's Code
        pshow = (pax*500).astype(np.int32)
        origin = np.asarray(cntr) #Np array

        cv2.line(self.img, tuple(origin+[50, 50]), tuple(pshow[:,0] + origin),\
                             (255,0,255),20, cv2.LINE_AA)
        cv2.line(self.img, tuple(origin), tuple(pshow[:,1] + origin),\
                             (0,255,255),20, cv2.LINE_AA)
#
        # Draw Center
        cv2.circle(self.img, cntr, 20, (0, 0, 0), 2)

        return angle
        
    
    def principal_axes(self):
        # Since we are in 2D, the principal axes are the max and min variance
        #   directions. I can use the mask boundary to find these, since the
        #   "mass distribution" of a mask is uniform. But I do need to remove
        #   the average value from the mask coordinates; SVD does not do that.
        #
        # The columns of the returned 2x2 array are unit vectors in the two principal
        #   directions, expressed in the camera coordinate system.
        #
        h, w, _ = self.mask.shape

        # we should agree on what a mask is.
        billmask = self.mask
        billmask = (self.mask[:,:,0].reshape(h, w, 1))
        billmask = (self.mask[:,:,0]).reshape((h,w,1))
        billmask[billmask > 0] = 1.0

        bpts = get_mask_boundary(billmask, self.xy_grid).astype(np.float32)
#        print('bpts shape is',bpts.shape)
        bpts = bpts - np.mean(bpts,axis=0,keepdims=True)
        p_axes, _,  _ = np.linalg.svd(bpts.transpose())

        return p_axes  # columns are the principal axes
    
        
class Micropipette(DetectedObject):
    def getTipPixel(self):
        print("Find")

##### General Functions #####

def drawPixelLocation(img, pixel, color=(255,0,0), radius=20, thickness=-1):
    cv2.circle(img, pixel, radius, color, thickness)
    
def displayImage(img):
    while True:
        cv2.imshow("Display", cv2.resize(img, (1280, 720)))
        k = cv2.waitKey(1)  # This pegs my CPU. 
        if k == 27:
            break
        
def get_mask_boundary(mask, xy_grid): # returns Nx2
    bmask = nd.binary_dilation(mask) - mask
    pts = mask_to_points(bmask, xy_grid)
    return pts

def mask_to_points(mask, xy_grid):  # returns Nx2
    xx, yy = xy_grid
    mgtz = np.squeeze(mask > 0, axis=2)
    xmask = xx[mgtz].reshape((1,-1))
    ymask = yy[mgtz].reshape((1,-1))
    pts = np.concatenate((xmask, ymask),axis=0)
    return pts.transpose()


##### Core #####

def identifyObjectInImage(img_path, mask_path):
    # Read original image and black and white mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    
    # Create Object
    micropipette = Micropipette(img, mask, [])
    micropipette.drawOutline()
    angle = micropipette.getOrientation()
    #print("Angle in radians: ", angle)
    #print("Angle in degrees: ", angle*(360/(2/pi)))
        
    displayImage(img)
 
if __name__=="__main__":
    identifyObjectInImage("Micropipette/Image.jpg", "Micropipette/Image_Mask.png")
    
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
