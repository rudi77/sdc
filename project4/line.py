# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:09:56 2017

@author: rudi
"""
import collections
import numpy as np

# Defines a line in a single frame. There shall be a left and a right line.
class LineSegment():
    def __init__(self, coeffs=None, fitx=None, x=None, y=None, vpos=None):
        # was the line in the current frame detected        
        self.detected = True if fitx is not None and coeffs is not None else False

        # coefficients of the fitted line
        self.coefficients = coeffs

        # x values of the fitted line
        self.xfitted = fitx
        
        # x indices of the line pixels
        self.x = x
        
        # y indices of the line pixels
        self.y = y
        
        if x is not None and y is not None:
            assert(len(x) == len(y))
            
        # radius of the curvature in meters
        self.radius = self.__calc_curvature__(self.xfitted) if self.xfitted is not None else None
        
        # contains the vehicle's position in meters
        self.vehicle_position = vpos
        
    def __calc_curvature__(self, xfitted):
        """
        Calculates the curvature of a line
        """
        ploty = np.linspace(0, 720-1, 720 )
        y_eval = np.max(ploty)
    
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720 # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
    
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * ym_per_pix, xfitted * xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])
        
        return round(curverad,1)

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, max_lines=3):
        self.__line_segments = collections.deque(maxlen=max_lines)
        
    def add_line(self, line_segment):
        self.__line_segments.append(line_segment)
        
    def get_last_line(self):
        return self.__line_segments[-1]
        
    def last_line_detected(self):
        if len(self.__line_segments) == 0:
            return False
        
        last_line = self.__line_segments[-1]

        return last_line.detected
    
    def is_new_line_valid(self, line_segment):
        """
        Validates line segment. It compares the line segment with the last
        inserted line segment.
        It compares the coefficients and the radis. +-10% deviation is ok but not
        more. If the last line segement is invalid (NONE) then this line segment
        is definitely valid.
        """
        if self.__line_segments[-1] is None:
            return True
        
        percent = self.__line_segments[-1].radius * (100. / line_segment.radius)
        radi_diff_percent = abs(100.0 - percent)
        
        return True if radi_diff_percent <= 10 else False
    
    def smoothed_coefficients(self):
        """
        Calcs average of coefficients of last n line segments
        """
        coeffs = []
        
        for ls in self.__line_segments:
            if ls is not None:
                coeffs.append(ls.coefficients)
        
        return np.mean(coeffs, axis=0)
    
    def reset(self):
        self.__line_segments.clear()
        
        