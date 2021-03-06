# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:59:40 2017

@author: rudi
"""
import os
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import helpers
import line

LeftLine = line.Line(max_lines=3)
RightLine = line.Line(max_lines=3)

img_counter = 0

def process_image(img, mtx, dist, M, Minv):
    global img_counter
    
    # undistort image
    img_undist = helpers.undistort(img, mtx, dist)
    
    # birds eye view
    img_birds_eye = helpers.warp2(img_undist, M)
        
    # binarize image
    b_channel = cv2.cvtColor(img_birds_eye, cv2.COLOR_RGB2LAB)[:,:,2]
    b_binary = helpers.binary_channel(b_channel, min_thresh=145, max_thresh=180)    
    g_channel = img_birds_eye[:,:,1]
    g_binary = helpers.binary_channel(g_channel, min_thresh=170, max_thresh=255)  
    
    # calc gradients in x direction
    gradx = helpers.gradx(img_birds_eye, min_thresh=80, max_thresh=120)
    
    # combine all binaries to a single binary image
    cg_binary = np.zeros_like(b_binary)
    cg_binary[(b_binary == 1) | (g_binary == 1) | (gradx == 1)] = 1
    
    is_valid_line = True
    # Do a blind search. A blind search is done at the very beginning or if we could not
    # find a lane with our improved search method
    if LeftLine.last_line_detected() == False or RightLine.last_line_detected() == False:
        LeftLineSegment, RightLineSegment = helpers.blind_search(cg_binary.copy())
        LeftLine.add_line(LeftLineSegment)
        RightLine.add_line(RightLineSegment)

    # improved search based on the previous result
    else:
        left_fit = LeftLine.get_last_line().coefficients
        right_fit = RightLine.get_last_line().coefficients
        LeftLineSegment, RightLineSegment = helpers.search_next(cg_binary.copy(), left_fit, right_fit)
        
        # Do a sanity check before we add the new line segment to the line
        if LeftLine.is_valid_line(LeftLineSegment) and RightLine.is_valid_line(RightLineSegment):
            LeftLine.add_line(LeftLineSegment)
            RightLine.add_line(RightLineSegment)
        else:
            is_valid_line = False

    # project the line on the undistorted image
    left_fitx = LeftLine.get_smoothed_line(num_frames=3)
    right_fitx = RightLine.get_smoothed_line(num_frames=3)
    result_img = helpers.project_lines(img_undist, cg_binary, left_fitx, right_fitx, Minv)
    
    # Get curvature radius, vehicle position and lane width
    left_radius = LeftLine.get_last_line().radius
    right_radius = RightLine.get_last_line().radius
    vehicle_position = RightLine.get_last_line().vehicle_position
    lane_width = LeftLine.get_last_line().lane_width

    helpers.print_measurements(result_img, left_radius, right_radius, vehicle_position, lane_width)
    
    # If lane finding failed, then reset left and right lines.
    if not is_valid_line:
        LeftLine.reset()
        RightLine.reset()
        
    if img_counter == 0:
        plt.imsave("./output_images/lane_visualization.jpg", result_img)
        img_counter += 1
    
    # return final image
    return result_img

def main():
    
    if not os.path.isfile('parameters.p'):
        # Load images for camera calibration
        cal_images = glob.glob("./camera_cal/calibration*.jpg")
        
        # Calibrate camera. Do this only once.
        mtx, dist = helpers.calibrate_camera(cal_images, nx=9, ny=5)
        
        # Get M and Minv
        img = mpimg.imread('./test_images/test1.jpg')
        img_undist = cv2.undistort(img, mtx, dist, None, mtx)
        warped_img, M = helpers.warp(img_undist)
        unwarped_img, Minv = helpers.unwarp(warped_img, img_undist.shape)
        
        persistables = { 'mtx' : mtx, 'dist' : dist, 'M' : M, 'Minv' : Minv }
        helpers.persist('parameters.p', persistables, './')
    else:
        params = helpers.load('parameters.p', './')
        mtx = params['mtx']
        dist = params['dist']
        M = params['M']
        Minv = params['Minv']
    
    
    # Ok, now we have M and Minv which we will use for image warping and unwarping in all frames
    process = lambda image: process_image(image, mtx, dist, M, Minv)
    
    #output = 'project_video_result.mp4'
    #clip1 = VideoFileClip("project_video.mp4")
    
    output = 'challenge_video_result.mp4'
    clip1 = VideoFileClip("challenge_video.mp4")
    
    #output = 'harder_challenge_video_result.mp4'
    #clip1 = VideoFileClip('harder_challenge_video.mp4')
    
    clip = clip1.fl_image(process)
    
    print("start video processing...")
    clip.write_videofile(output, audio=False)
    
if __name__ == "__main__": main()
    
