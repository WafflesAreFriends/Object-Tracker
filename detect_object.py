import cv2
import imutils
import numpy as np
from math import atan2, cos, sin, sqrt, pi, hypot
from scipy import ndimage as nd

# def get_mask_boundary(mask, xy_grid): # returns Nx2
#     bmask = nd.binary_dilation(mask) - mask
#     pts = mask_to_points(bmask, xy_grid)
#     return pts

def get_mask_boundary(mask, xy_grid): # returns Nx2
#    bmask = nd.binary_dilation(mask) - mask
#    pts = mask_to_points(bmask, xy_grid)
#    return pts

#Bill's from points_util.py as of 6/30/20
    c = None
#    mask[nd.binary_dilation(mask)]=1
    contours, hierarchy = \
        cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours)==0:
        print('no contours?')
        return c
#
# Use longest contour, if more than one is found.
#
    lmax=0
    ipick = 0
    for i,c in enumerate(contours):
        n = c.shape[0]
        if n > lmax:
            lmax=n
            ipick=i

    c = contours[ipick]      
    return c.squeeze()

def mask_to_points(mask, xy_grid):  # returns Nx2
    xx, yy = xy_grid
    mgtz = np.squeeze(mask > 0, axis=2)
    xmask = xx[mgtz].reshape((1,-1))
    ymask = yy[mgtz].reshape((1,-1))
    pts = np.concatenate((xmask, ymask),axis=0)
    return pts.transpose()

##### Classes #####

class DetectedObject():
    def __init__(self, img, mask):
        self.h, self.w, _ = img.shape
        self.xy_grid = np.meshgrid(np.arange(self.w), np.arange(self.h)) # Bench
        self.update_object(img, mask)
        self.tracking = True

    def update_object(self, img, mask):
        self.img = img
        self.mask = mask
        
        self.mask_boundary = get_mask_boundary(self.mask, self.xy_grid).astype(np.float32)
        self.centroid = self.get_centroid()
        self.axes = self.get_principal_axes()

    # Returns numpy array of axes unit vectors
    def get_principal_axes(self):
        # Since we are in 2D, the principal axes are the max and min variance
        #   directions. I can use the mask boundary to find these, since the
        #   "mass distribution" of a mask is uniform. But I do need to remove
        #   the average value from the mask coordinates; SVD does not do that.
        #
        # The columns of the returned 2x2 array are unit vectors in the two principal
        #   directions, expressed in the camera coordinate system.
        #
        bpts = get_mask_boundary(self.mask, self.xy_grid).astype(np.float32)
        bpts = bpts - np.mean(bpts,axis=0,keepdims=True)
        p_axes, _,  _ = np.linalg.svd(bpts.transpose())
    
        return p_axes  # Columns are the principal axes
    
    # TODO: Find way to get angle from rotated coordinate system
    #def get_orientation(self):
        #angle = atan2(self.boundary) # orientation in radians
    
    def get_centroid(self):
        x0 = np.mean(self.mask_boundary, axis=0)
        return x0[0], x0[1]
        
    def get_dimensions(self):
        bpts = get_mask_boundary(self.mask, self.xy_grid) # Nx2
        ppts = bpts.dot(self.axes) # project boundary into principal coords
        length = np.max(ppts[:,0]) - np.min(ppts[:,0])
        width = np.max(ppts[:,1]) - np.min(ppts[:,1])
        return length, width
        
    # CV2 drawing is in BGR instead of RGB
    def draw_axes(self, x_axes_color = (0, 0, 255), y_axes_color = (255, 0, 0), thickness=20):
        x_axes_point = self.centroid+self.axes[0]*-1000
        y_axes_point = self.centroid+self.axes[1]*1000
    
        cv2.line(self.img, tuple(self.centroid), tuple(x_axes_point), x_axes_color, thickness, cv2.LINE_AA)
        cv2.line(self.img, tuple(self.centroid), tuple(y_axes_point), y_axes_color, thickness, cv2.LINE_AA)
    
    def draw_boundary(self, color = (0, 255, 0)):
        for pixel_loc in self.mask_boundary:
            self.img[int(pixel_loc[1]), int(pixel_loc[0])] = np.asarray(color)


class Micropipette(DetectedObject):
    def __init__(self, img, mask):
        super().__init__(img, mask)
#        self.has_tip = self.tip_on()

    def tip_direction(self):
        pa = self.axes  # returns princial axes in descending variance order
        maxvar_line = pa[:,0].transpose()
        minvar_line = pa[:.1].transpose()

        pts = mask_to_points(self.mask, self.xy_grid)
        # transform to principal coords
        x_along = maxvar_line.dot(pts)
        x_across = minvar_line.dot(pts)

        end0 = x_along < np.quantile(x_along, 0.2)
        end1 = x_along > np.quantile(x_along, 0.8)

        # Pick the narrow end and return unit vector that points toward it
        if mad(x_across[end1]) < mad(x_across[end0]):
            return maxvar_line
        else:
            return -maxvar_line

#    def tip_on(self):
#        TIP_CLASS_ID = 42 # this is unlikely to be correct
#        tip_on = False
#
#        # look for a tip adjacent to self in tip_direction()
#        tip_dir = self.tip_direction()
#
#        for obj in self.bench.objects_in_view:
#            if obj.classname == TIP_CLASS_ID:
#                x1, y1 = obj.centroid()
#                x0, y0 = self.centroid()
#                dvec = np.asarray([x1-x0, y1-y0])
#                dist = np.sqrt(np.pow(dvec,2))
#                if dist < self.dimensions[0]/2:
#                    if tip_dir.dot(dvec)/dist > 0.98 : # within about 20 degrees
#                        tip_on = True
#                        break
#
#        return tip_on

class PCR_Plate():
    pass

class Sticker(DetectedObject):
    def __init__(self, img, mask):
        super().__init__(img, mask) 

    # Checks if sticker overlaps inputted mask
    def check_mask_overlap(self, obj_mask):
        mask_intersect = np.bitwise_and(self.mask, obj_mask)
        
        if np.array_equal(mask_intersect, self.mask):
            return True 

        return False
    

class Rack(DetectedObject):
    pass

##### General Functions #####

def draw_pixel_location(img, pixel, color=(255,0,0), radius=20, thickness=-1):
    cv2.circle(img, pixel, radius, color, thickness)
    
def display_image(img, scale=1):
    h, w, _ = img.shape
    while True:
        scaled_width = w*scale
        scaled_height = h*scale
        cv2.imshow("Display", cv2.resize(img, (scaled_width, scaled_height)))
        k = cv2.waitKey(1)  # This pegs my CPU. 
        if k == 27:
            break
    cv2.destroyAllWindows()

##### Core #####

# Turns png image to yolact like mask
# If want to display mask, turn all 1 values to RGB channel
def get_mask_from_image(mask):
    h, w, _ = mask.shape
    mask = (mask[:,:,0].reshape(h, w, 1))
    mask = (mask[:,:,0]).reshape((h,w,1))
    mask[mask > 0.5] = 1.0
    return mask

def identify_object(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = get_mask_from_image(cv2.imread(mask_path))
    
    # Create Micropipette Object
    micropipette = Micropipette(img, mask)
    micropipette.draw_axes()
    micropipette.draw_boundary()
    
    display_image(img)

# Checks if a sticker is on an object
def confirm_sticker_on_obj(img_path, obj_mask_path, sticker_mask_path):
    # Get masks and images
    img = cv2.imread(img_path)
    obj_mask = get_mask_from_image(cv2.imread(obj_mask_path))
    sticker_mask = get_mask_from_image(cv2.imread(sticker_mask_path))

    # Create objects
    micropipette = Micropipette(img, obj_mask)
    micropipette.draw_boundary()
    sticker = Sticker(img, sticker_mask)
    sticker.draw_boundary()

    on_mask = False
    if sticker.check_mask_overlap(obj_mask):
        print("Sticker is on the object :)")
        on_mask = True
    else:
        print ("Sticker is not on the object :(")

    #display_image(img)
    return on_mask

# this function is a work in progress so it's a bit of a mess right now, but it will
# eventually check the number of tubes in a rack
def confirm_tubes_in_rack(correct_num_tubes, img_path, rack_mask_path, sticker_mask_paths):
    img = cv2.imread(img_path)
    rack_mask = get_mask_from_image(cv2.imread(rack_mask_path))
    rack = Rack(img, rack_mask)

    sticker_mask_0 = get_mask_from_image(cv2.imread(sticker_mask_paths[1]))
    sticker_0 = Sticker(img, sticker_mask_0)

    sticker_x = sticker_0.centroid[0]
    sticker_y = sticker_0.centroid[1]
    # print(sticker_x)
    # print(sticker_y)
    # print(sticker_0.centroid)

    length, width = rack.get_dimensions()
    # print(length)
    # print(width)

    x_axes_color = (0, 0, 255)
    y_axes_color = (255, 0, 0)
    thickness=20
    x_axes_point = rack.centroid+rack.axes[0] * (length / 2)
    x_axes_point_2 = rack.centroid-rack.axes[0] * (length / 2)
    y_axes_point = rack.centroid+rack.axes[1] * (width / 2)
    y_axes_point_2 = rack.centroid-rack.axes[1] * (width / 2)

    # a = x_axes_point[1] - rack.centroid[1] 
    # b = rack.centroid[0] - x_axes_point[0]  
    # c = -1 * (a*(rack.centroid[0]) + b*(rack.centroid[1]))
    # d = abs((a * sticker_x + b * sticker_y + c)) / (sqrt(a * a + b * b))    

    # calculate distance and point on principal axis
    x1, y1 = x_axes_point_2
    x2, y2 = x_axes_point
    x3, y3 = sticker_0.centroid
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    final_point = x1+a*dx, y1+a*dy

    dist = hypot(final_point[0] - sticker_x, final_point[1] - sticker_y)
    print("The distance is: ", dist)

    # distance
    cv2.line(img, tuple(final_point), tuple(sticker_0.centroid), (0, 255, 0), thickness, cv2.LINE_AA)

    #principal axes
    cv2.line(img, tuple(x_axes_point_2), tuple(x_axes_point), x_axes_color, thickness, cv2.LINE_AA)
    cv2.line(img, tuple(y_axes_point_2), tuple(y_axes_point), y_axes_color, thickness, cv2.LINE_AA)
    
    #rack.draw_axes()  
    display_image(img)

# MAIN 
if __name__=="__main__":
    #identify_object("Sticker/Image.jpg", "Sticker/Pipette_Mask.png")
    #identify_object("PCR_Plate/Image.jpg", "PCR_Plate/Image_Mask.png")
    confirm_sticker_on_obj("Sticker/Image.jpg", "Sticker/Pipette_Mask.png", "Sticker/Sticker_Mask.png")
    confirm_sticker_on_obj("Sticker/Image.jpg", "Sticker/Pipette_Mask.png", "Sticker/Incorrect_Sticker_Mask.png")
    #sticker_mask_paths = ["Sticker/Rack/H2O_Mask_1.png", "Sticker/Rack/N1_Mask_1.png", "Sticker/Rack/N2_Mask_1.png", "Sticker/Rack/PC_Mask_1.png", "Sticker/Rack/PN_Mask_1.png", "Sticker/Rack/RP_Mask_1.png", "Sticker/Rack/S_Mask_1.png"]
    #confirm_tubes_in_rack(5, "Sticker/Rack/Rack_Image_1.jpg", "Sticker/Rack/Rack_Mask_1.png", sticker_mask_paths)
