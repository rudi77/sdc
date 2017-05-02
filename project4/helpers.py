# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:22:29 2017

@author: rudi

"""
import os
import pickle
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import line


def persist(filename, persistable_data, path):
    
    file = os.path.join(path, filename)
    
    # Save the data for easy access
    if not os.path.isfile(file):
        print('Saving data to pickle file...')
        try:
            with open(file, 'wb') as pfile:
                pickle.dump(persistable_data, pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    
    print('Data cached in pickle file.')    
    
def load(filename, path):
    """
    Loads images and labels from a pickle file.
    """
    file = os.path.join(path, filename)
    
    try:
        with open(file, mode='rb') as pfile:
            return pickle.load(pfile)
    except Exception as e:
        print('Unable to open file', file, ':', e)
        raise

############### Camera Calibration ############################################

def calibrate_camera(cal_images, nx, ny):
    # arrays to store all object points and image points from all images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # prepare object points like (0,0,0), (1,0,0),(2,0,0) ... (8,4,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates
    
    img_shape = None
    
    for img_name in cal_images:
        img = mpimg.imread(img_name)
        
        if img_shape is None:
            img_shape = (img.shape[1], img.shape[0])
        
        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # find chessboardcorners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    assert(len(objpoints) == len(imgpoints))
    
    # get camera calibration matrix and distoration coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    
    return mtx, dist

############### Image Undistortion ############################################
    
def undistort(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)

############### Perspective Transformation ####################################

def warp(undist):
    src = np.float32([[710,460],[1110,720],[205,720],[575,460]])
    dst = np.float32([[1000,0],[1000,720],[300,720],[300,0]])    
        
    M = cv2.getPerspectiveTransform(src, dst)
    
    img_size = (undist.shape[1], undist.shape[0])
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M

def warp2(undist, M):
    img_size = (undist.shape[1], undist.shape[0])    
    warped = cv2.warpPerspective(undist, M, img_size)          
    return warped

############### Inverse Perspective Transformation ############################

def unwarp(birds_eye_view, shape):
    src = np.float32([[1000,0],[1000,720],[300,720],[300,0]]) 
    dst = np.float32([[710,460],[1110,720],[205,720],[575,460]])
        
    Minv = cv2.getPerspectiveTransform(src, dst)
    
    img_size = (shape[1], shape[0])
    unwarped = cv2.warpPerspective(birds_eye_view, Minv, img_size)      
    return unwarped, Minv

def unwarp2(birds_eye_view, Minv, shape):    
    img_size = (shape[1], shape[0])
    unwarped = cv2.warpPerspective(birds_eye_view, Minv, img_size) 
    return unwarped

############### Color and Gradient Transformation #############################

def binary_b_channel(img, min_thresh=145, max_thresh=180):
    b_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:,:,2]
    b_binary = np.zeros_like(b_img)
    b_binary[(b_img >= min_thresh) & (b_img <= max_thresh)] = 1
    return b_binary

def binary_v_channel(img, min_thresh=225, max_thresh=255):
    v_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2]
    v_binary = np.zeros_like(v_img)
    v_binary[(v_img >= min_thresh) & (v_img <= max_thresh)] = 1
    return v_binary

def binary_channel(img, min_thresh=0, max_thresh=255, ):
    """
    A generic binary image function.
    """
    binary = np.zeros_like(img)
    binary[(img >= min_thresh) & (img <= max_thresh)] = 1
    return binary

def gradx(img, kernel=11, min_thresh=50, max_thresh=100):
    # calculates derivatives in x direction:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2]
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    
    # creates a binary images based on the provided min and max thresholds
    sbinary[(scaled_sobel >= min_thresh) & (scaled_sobel <= max_thresh)] = 1
    return sbinary

########################### Finding Lanes #####################################
def polyfit(x, y):
    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)
    
    # Generate x and y values for plotting
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    ploty = np.linspace(0, 720-1, 720 )
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    
    return fit, fitx

def calc_vehicle_pos(leftx, rightx, midpoint):
    meters_per_pixel = 3.7 / 700.0
    lane_width = round((rightx - leftx) * meters_per_pixel, 1)
        
    # lane midpoint
    lane_center = leftx + int((rightx - leftx) / 2)
    # calculate difference between lane midpoint and image midpoint which
    # is the deviation of the car to the lane midpoint
    diff = abs(lane_center - midpoint)    
    deviation = round(diff * meters_per_pixel, 2)
    
    return deviation, lane_width

def blind_search(binary_warped):
    # crop the image. The last 20 pixel won't be taken into account because
    # these pixels belong to the engine hood and pixels which belong to it
    # shall not influence line finding
    binary_warped = binary_warped[0:700,:]
    
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)    
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    vehicle_pos, lane_width = calc_vehicle_pos(leftx_base, rightx_base, midpoint)
    
    # Choose the number of sliding windows. I use 10 because 700 can be divided by 10
    nwindows = 10
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    # numpy.nonzero(): returns a tuple of arrays, one for each dimension, 2 arrays in this case.
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Create Left and Right Line Segments    
    left_fit, left_fitx = polyfit(leftx, lefty)
    LeftLineSegment = line.LineSegment(left_fit, left_fitx, leftx, lefty, vehicle_pos, lane_width)
    
    right_fit, right_fitx = polyfit(rightx, righty)
    RightLineSegment = line.LineSegment(right_fit, right_fitx, rightx, righty, vehicle_pos, lane_width)

    return LeftLineSegment, RightLineSegment


def search_next(binary_warped, left_fit, right_fit):
    """
    @binary_warped  warped binary image of this frame
    @left_fit    coefficients of the left poly fit of the previous frame
    @right_fit   coefficients of the right poly fit of the previous frame
    Searches for the lines based on the result of the previous frame.
    """
    binary_warped = binary_warped[0:700,:]
       
    # Get all nonzero pixels from the warped image.
    # Store the x and y indices in separate lists
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    margin = 100
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Create Left and Right Line Segments    
    left_fit, left_fitx = polyfit(leftx, lefty)
    right_fit, right_fitx = polyfit(rightx, righty)
    
    midpoint = np.int(binary_warped.shape[1] / 2)    
    vehicle_pos, lane_width = calc_vehicle_pos(left_fitx[-1], right_fitx[-1], midpoint)
    
    LeftLineSegment = line.LineSegment(left_fit, left_fitx, leftx, lefty, vehicle_pos, lane_width)
    RightLineSegment = line.LineSegment(right_fit, right_fitx, rightx, righty, vehicle_pos, lane_width)

    return LeftLineSegment, RightLineSegment
    

def project_lines(undist, warped, left_fitx, right_fitx, Minv):
    ploty = np.linspace(0, 720-1, 720 )
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result


def print_measurements(img, left_radius, right_radius, vehicle_position, lane_width):
    # put radius and vehicle position on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Left radius {} m'.format(left_radius)
    cv2.putText(img, text, (30,80), font, 1.2, (200,255,155), 4, cv2.LINE_AA)
    text1 = 'Right radius {} m'.format(right_radius)
    cv2.putText(img, text1, (30,120), font, 1.2, (200,255,155), 4, cv2.LINE_AA)
    text2 = 'Vehicle position {} m'.format(vehicle_position)
    cv2.putText(img, text2, (30,160), font, 1.2, (200,255,155), 4, cv2.LINE_AA)
    text3 = 'Lane width {} m'.format(lane_width)
    cv2.putText(img, text3, (30,200), font, 1.2, (200,255,155), 4, cv2.LINE_AA)

