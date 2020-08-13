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

    def set_type(self, type):
        self.type = type

    # Checks if sticker overlaps inputted mask
    def check_mask_overlap(self, obj_mask):
        mask_intersect = np.bitwise_and(self.mask, obj_mask)
        
        if np.array_equal(mask_intersect, self.mask):
            return True 

        return False
    

class Rack(DetectedObject):
    def __init__(self, img, mask):
        super().__init__(img, mask) 

    def find_tube_dist_x(self, img, sticker):
        sticker_x = sticker.centroid[0]
        sticker_y = sticker.centroid[1]

        length, width = self.get_dimensions()

        x_axes_color = (0, 0, 255)
        thickness = 20
        x_axes_point = self.centroid + self.axes[0] * (length / 2)
        x_axes_point_2 = self.centroid - self.axes[0] * (length / 2)

        # calculate distance and point on principal axis
        x1, y1 = x_axes_point_2
        x2, y2 = x_axes_point
        x3, y3 = sticker.centroid
        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy
        a = (dy*(y3-y1)+dx*(x3-x1))/det
        final_point = x1+a*dx, y1+a*dy

        dist = hypot(final_point[0] - sticker_x, final_point[1] - sticker_y)
        #print("The distance (x) is: ", dist)

        # distance
        #cv2.line(img, tuple(final_point), tuple(sticker.centroid), (0, 255, 0), thickness, cv2.LINE_AA)

        #principal axis
        cv2.line(img, tuple(x_axes_point_2), tuple(x_axes_point), x_axes_color, thickness, cv2.LINE_AA)

        return dist

    def find_tube_dist_y(self, img, sticker):
        sticker_x = sticker.centroid[0]
        sticker_y = sticker.centroid[1]

        length, width = self.get_dimensions()

        y_axes_color = (255, 0, 0)
        thickness = 20
        y_axes_point = self.centroid + self.axes[1] * (width / 2)
        y_axes_point_2 = self.centroid - self.axes[1] * (width / 2)   

        # calculate distance and point on principal axis
        x1, y1 = y_axes_point_2
        x2, y2 = y_axes_point
        x3, y3 = sticker.centroid
        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy
        a = (dy*(y3-y1)+dx*(x3-x1))/det
        final_point = x1+a*dx, y1+a*dy

        dist = hypot(final_point[0] - sticker_x, final_point[1] - sticker_y)
        #print("The distance (y) is: ", dist)

        # distance
        #cv2.line(img, tuple(final_point), tuple(sticker.centroid), (0, 255, 0), thickness, cv2.LINE_AA)

        #principal axis
        cv2.line(img, tuple(y_axes_point_2), tuple(y_axes_point), y_axes_color, thickness, cv2.LINE_AA)

        return dist

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

    tube_count = 0
    dist_x = 0
    dist_y = 0

    for s in sticker_mask_paths:
        sticker_mask = get_mask_from_image(cv2.imread(s))
        sticker = Sticker(img, sticker_mask)
        dist_x = rack.find_tube_dist_x(img, sticker)
        dist_y = rack.find_tube_dist_y(img, sticker)
        if (dist_x <= 490 and dist_y <= 1400):
            tube_count += 1
        

    if (tube_count == correct_num_tubes):
        print("Correct number of tubes on rack!")
    elif (tube_count > correct_num_tubes):
        print("%d%s" % (tube_count - correct_num_tubes, " extra tubes on rack"))
    else:
        print("%s%d%s" % ("Missing ", correct_num_tubes - tube_count, " tubes from rack"))

    #rack.draw_axes()  
    display_image(img)



# MAIN 
if __name__=="__main__":
    identify_object("Sticker/Image.jpg", "Sticker/Pipette_Mask.png")
    #identify_object("PCR_Plate/Image.jpg", "PCR_Plate/Image_Mask.png")
    #confirm_sticker_on_obj("Sticker/Image.jpg", "Sticker/Pipette_Mask.png", "Sticker/Sticker_Mask.png")
    #confirm_sticker_on_obj("Sticker/Image.jpg", "Sticker/Pipette_Mask.png", "Sticker/Incorrect_Sticker_Mask.png")
    sticker_mask_paths_1 = ["Rack/1/Sticker_Mask_1.png", "Rack/1/Sticker_Mask_2.png", "Rack/1/Sticker_Mask_3.png", "Rack/1/Sticker_Mask_4.png", "Rack/1/Sticker_Mask_5.png", "Rack/1/Sticker_Mask_6.png", "Rack/1/Sticker_Mask_7.png"]
    #confirm_tubes_in_rack(7, "Rack/1/Rack_Image.jpg", "Rack/1/Rack_Mask.png", sticker_mask_paths_1)
    
    sticker_mask_paths_2 = ["Rack/2/Sticker_Mask_1.png", "Rack/2/Sticker_Mask_2.png", "Rack/2/Sticker_Mask_3.png", "Rack/2/Sticker_Mask_4.png", "Rack/2/Sticker_Mask_5.png", "Rack/2/Sticker_Mask_6.png", "Rack/2/Sticker_Mask_7.png"]
    #confirm_tubes_in_rack(7, "Rack/2/Rack_Image.jpg", "Rack/2/Rack_Mask.png", sticker_mask_paths_2)
    
    sticker_mask_paths_3 = ["Rack/3/Sticker_Mask_1.png", "Rack/3/Sticker_Mask_2.png", "Rack/3/Sticker_Mask_3.png", "Rack/3/Sticker_Mask_4.png", "Rack/3/Sticker_Mask_5.png", "Rack/3/Sticker_Mask_6.png", "Rack/3/Sticker_Mask_7.png"]
    #confirm_tubes_in_rack(7, "Rack/3/Rack_Image.jpg", "Rack/3/Rack_Mask.png", sticker_mask_paths_3)

    sticker_mask_paths_4 = ["Rack/4/Sticker_Mask_1.png", "Rack/4/Sticker_Mask_2.png", "Rack/4/Sticker_Mask_3.png", "Rack/4/Sticker_Mask_4.png", "Rack/4/Sticker_Mask_5.png", "Rack/4/Sticker_Mask_6.png", "Rack/4/Sticker_Mask_7.png"]
    #confirm_tubes_in_rack(7, "Rack/4/Rack_Image.jpg", "Rack/4/Rack_Mask.png", sticker_mask_paths_4)
