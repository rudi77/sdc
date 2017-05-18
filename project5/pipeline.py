# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:55:11 2017

@author: rudi
"""
import os
import collections
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import itertools
from sklearn import metrics
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from scipy.ndimage.measurements import label

import helpers
import line
import dataset_generator as dg


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, classifier, x_scaler, scale=1, ystart=400, ystop=656, orient=6, pix_per_cell=8, cells_per_block=2, colors=[(0,255,0), (255,0,0)]):
    
    draw_img = np.copy(img)    
    img_tosearch = img[ystart:ystop,:,:]
    
    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    img_converted = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    
    # Define blocks and steps as above
    nxblocks = (img_converted.shape[1] // pix_per_cell) - cells_per_block + 1
    nyblocks = (img_converted.shape[0] // pix_per_cell) - cells_per_block + 1 
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cells_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image  
    hog1 = dg.hog_features(img_converted[:,:,0], orient, pix_per_cell, cells_per_block)    
    hog2 = dg.hog_features(img_converted[:,:,1], orient, pix_per_cell, cells_per_block)
    hog3 = dg.hog_features(img_converted[:,:,2], orient, pix_per_cell, cells_per_block)
            
    # contains all boxes which may contain a vehicle
    boxes = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
                        
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            subimg = img_converted[ytop:ytop+window, xleft:xleft+window]
                       
            # Extract HOG for this patch
            hog_features_1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features_2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features_3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            
            # Extract color features                              
            col_features = dg.color_hist(subimg, nbins=16)
            spat_features = dg.bin_spatial(subimg, size=(16, 16))

            # predict
            X = np.concatenate((hog_features_1, hog_features_2, hog_features_3, col_features, spat_features))            
            X = [np.array(X).astype(np.float64)]            
            scaled_features = x_scaler.transform(X)        
            test_prediction = classifier.predict(scaled_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)                
                boxes.append([(xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)])                
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),colors[0],2)
            #else:
            #    xbox_left = np.int(xleft*scale)
            #    ytop_draw = np.int(ytop*scale)
            #    win_draw = np.int(window*scale)
            #    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),colors[1],2)


    return draw_img, boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image        
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


LeftLine = line.Line(max_lines=3)
RightLine = line.Line(max_lines=3)
img_counter = 0

def detect_lane(img, mtx, dist, M, Minv):
    global img_counter
    global LeftLine
    global RightLine
    
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
        
def detect_cars(img, clf, x_scaler, box_deque, search_windows):    
    img_copy = img.copy().astype(np.float32)/255
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    
    box_list = []
    images_with_boxes = []
    
    for sw in search_windows:
        img_hog, boxes = find_cars(img_copy, clf, x_scaler, scale=sw[0], ystart=sw[1][0], ystop=sw[1][1], orient=9,  colors=sw[3])
                
        if len(boxes) > 0:
            box_list.extend(boxes)
        images_with_boxes.append(img_hog)
        
    box_deque.append(box_list)
    
    # Add heat to each box in box list
    all_boxes = []
    for boxes in box_deque:
        all_boxes.extend(boxes)
        
    heat = add_heat(heat,all_boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,5)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)    
    #histogram = np.sum(heatmap, axis=0)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)    
    
    plot = False
    show_all = False
    
    if plot == True:
        fig = plt.figure(figsize=(20,8))

        pos = 241
        if show_all:
            plt.subplot(pos)
            plt.imshow(img)
            plt.title(img_name)
            
            pos += 1
            
            cnt = 0
            for iwb in images_with_boxes:            
                plt.subplot(pos)
                plt.imshow(iwb)
                plt.title(search_windows[cnt][2])
                cnt += 1
                pos += 1
                        
            plt.subplot(pos)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heatmap')
            pos += 1
            
            plt.subplot(pos)
            plt.imshow(draw_img)
            plt.title('All boxes')
            fig.tight_layout()
        else: 
            plt.subplot(131)
            plt.imshow(img)
            plt.title(img_name)
            plt.subplot(132)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heatmap')
            plt.subplot(133)
            plt.imshow(draw_img)
            plt.title('All boxes')
            fig.tight_layout()                    
    
    return draw_img

def process_image(image, mtx, dist, M, Minv, clf, x_scaler, box_deque, search_windows):
    img = detect_cars(image, clf, x_scaler, box_deque, search_windows)
    img = detect_lane(img, mtx, dist, M, Minv)
    
    return img


def main():
    # load the params needed for image un-distortion
    params = helpers.load('parameters.p', './')
    mtx = params['mtx']
    dist = params['dist']
    M = params['M']
    Minv = params['Minv']
    
    # Load svm model and scaler
    model = joblib.load('svm_dataset_8_2_9_3_32.pkl')
    clf = model['clf']
    x_scaler = model['scaler']
    
    # stores boxes of the last n frames
    box_deque = collections.deque(maxlen=10)
    
    #search_windows = [(1, [400, 500],  "Img 64x64", [(0,255,0), (255,0,0)]),
    #                   (1.5, [400, 656],  "Img 96x96", [(0,255,0), (200,200,0)]),
    #                   (2,   [450, None],  "Img 128x128",[(0,255,0), (120,120,0)]),
    #                   (2.5, [450, None], "Img 160x160",[(0,255,0), (80,120,160)])]  # 160x160
    
    search_windows = [(1.5, [400, 656],  "Img 96x96", [(0,255,0), (200,200,0)]),
                      (2,   [450, None],  "Img 128x128",[(0,255,0), (120,120,0)]),
                      (2.5, [450, None], "Img 160x160",[(0,255,0), (80,120,160)])]  # 160x160    
        
    process = lambda image: process_image(image, mtx, dist, M, Minv, clf, x_scaler, box_deque, search_windows)

    output = 'project_video_processed.mp4'
    clip = VideoFileClip("project_video.mp4")    
    #output = 'test_video_processed2.mp4'
    #clip = VideoFileClip("test_video.mp4")    
    output_clip = clip.fl_image(process) 
    output_clip.write_videofile(output, audio=False)

#test_images = glob.glob('test_images/*.jpg')

#for img_name in test_images:
    #img = mpimg.imread(img_name)
    #process_image(img, plot=True, convert=False)
    #mpimg.imsave(img_name, img)

if __name__ == "__main__": main()