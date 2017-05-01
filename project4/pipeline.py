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
from moviepy.editor import VideoFileClip
import helpers
import line

LeftLine = line.Line()
RightLine = line.Line()


def process_image(img, mtx, dist, M, Minv):
    # undistort image
    img_undist = helpers.undistort(img, mtx, dist)
    
    # birds eye view
    img_birds_eye = helpers.warp2(img_undist, M)
        
    # binarize image
    b_binary = helpers.binary_b_channel(img_birds_eye)    
    v_binary = helpers.binary_v_channel(img_birds_eye)  
    
    # calc gradients in x direction
    gradx = helpers.gradx(img_birds_eye)
    
    # combine all binaries to a single binary image
    cg_binary = np.zeros_like(gradx)
    cg_binary[(b_binary == 1) | (v_binary == 1) | (gradx == 1)] = 1
        
    # do a blind search
    if LeftLine.last_line_detected() == False or RightLine.last_line_detected() == False:
        LeftLineSegment, RightLineSegment = helpers.blind_search(cg_binary.copy())
        LeftLine.add_line(LeftLineSegment)
        RightLine.add_line(RightLineSegment)

    # improved search based on the previous result
    else:
        left_fit = LeftLine.get_last_line().coefficients
        right_fit = RightLine.get_last_line().coefficients
        LeftLineSegment, RightLineSegment = helpers.search_next(cg_binary.copy(), left_fit, right_fit)

    # project the line on the undistorted image
    left_fitx = LeftLineSegment.xfitted
    right_fitx = RightLineSegment.xfitted
    result_img = helpers.project_lines(img_undist, cg_binary, left_fitx, right_fitx, Minv)

    # put radius and vehicle position on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Left radius {} m'.format(LeftLineSegment.radius)
    cv2.putText(result_img, text, (30,80), font, 1.2, (200,255,155), 4, cv2.LINE_AA)
    text = 'Right radius {} m'.format(RightLineSegment.radius)
    cv2.putText(result_img, text, (30,120), font, 1.2, (200,255,155), 4, cv2.LINE_AA)
    text2 = 'Vehicle position {} m'.format(LeftLineSegment.vehicle_position)
    cv2.putText(result_img, text2, (30,160), font, 1.2, (200,255,155), 4, cv2.LINE_AA)
    
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
    
    output = 'project_video_result4.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process)
    
    print("start video processing...")
    white_clip.write_videofile(output, audio=False)
    
if __name__ == "__main__": main()
    
