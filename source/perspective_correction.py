import os
import math
import cv2
import numpy as np
from PIL import Image

def mask_convex_hull(mask):
    # Find object with the maximum convex hull and draw it on the image

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
    
    contour_label = np.zeros_like(mask)
    
    cv2.drawContours(contour_label, contours, 0, 255, 1)

    return contour_label

def find_hesse_intersection(line_1, line_2):
    # xcos(t1) + ysin(t1) = r1
    # xcos(t2) + ysin(t2) = r2
    
    # or
    
    # [ cos(t1) sin(t1) ] [x] = [r1]
    # [ cos(t2) sin(t2) ] [y] = [r2]
    
    A = np.array([
        [np.cos(line_1[1]), np.sin(line_1[1])],
        [np.cos(line_2[1]), np.sin(line_2[1])]
    ])
        
    b = np.matrix([[line_1[0], line_2[0]]]).T
    
    res = np.linalg.solve(A, b)
    
    x = res[0, 0]
    y = res[1, 0]
    
    return x, y

def find_quadrilateral_points(label, intersection_tolerance=0.05, same_multiplier=1.0):
    convex_hull = mask_convex_hull(label)
    
    lines = cv2.HoughLines(convex_hull, rho=1, theta=np.pi/720, threshold=10)
    
    # Find four of the most rated lines that are sufficiently 
    # distinct
    
    current_lines = []

    rho_tol = 20*same_multiplier
    angle_tol = np.pi/10*same_multiplier
    
    for rho, theta in lines.squeeze():        
        invalidate = False
        for existing_rho, existing_theta in current_lines:
            # Pass on lines that are similar to already discovered
            if abs(rho-existing_rho) < rho_tol and abs(theta-existing_theta) < angle_tol:
                invalidate = True
                break
            # Also check for similarity in the "discontinuity" of the transform
            if abs(abs(rho)-abs(existing_rho)) < rho_tol and np.pi-angle_tol < abs(abs(theta)-abs(existing_theta)) < np.pi+angle_tol:
                invalidate = True
                break
        if not invalidate:
            current_lines.append((rho, theta))
            
        if len(current_lines) == 4:
            break
            
    points = []
    
    abs_tol = intersection_tolerance*label.shape[0]+intersection_tolerance*label.shape[1]

            
    for l1_idx, l1 in enumerate(current_lines):
        for l2 in current_lines[l1_idx+1:]:
            try:
                i_x, i_y = find_hesse_intersection(l1, l2)
            except:
                continue
            
            if i_x < label.shape[1]+abs_tol and i_x >= -abs_tol and i_y < label.shape[0]+abs_tol and i_y >= -abs_tol:
                points.append({'x': i_x, 'y': i_y})
    return points
        
def correct_perspective(image, label):
    tr = None
    for n_retries in range(1, 4):
        quad_points = find_quadrilateral_points(label, same_multiplier=n_retries)
        
        def center_angle(point, ref_y, ref_x):
            px = point['x']-ref_x
            py = point['y']-ref_y
            return np.arctan2(py, px)

        
        center_x = sum(point['x'] for point in quad_points)/4
        center_y = sum(point['y'] for point in quad_points)/4
        
        
        target_points = [
            {'x': image.shape[1], 'y': 0},
            {'x': 0, 'y': 0},
            {'x': 0, 'y': image.shape[0]},
            {'x': image.shape[1], 'y': image.shape[0]},
        ]  
        
        target_points = sorted(target_points, key=lambda point: center_angle(point, center_y, center_x))
        quad_points = sorted(quad_points, key=lambda point: center_angle(point, center_y, center_x))
        
        target_mat = np.matrix([[p['x'], p['y']] for p in target_points], np.float32)
        dst_mat = np.matrix([[p['x'], p['y']] for p in quad_points], np.float32)
        
        try:
            tr = cv2.getPerspectiveTransform(
                src = target_mat,
                dst = dst_mat
            )
            break
        except:
            continue
    
    inv_tr = np.linalg.inv(tr)
    
    return inv_tr, cv2.warpPerspective(image, inv_tr, (image.shape[1], image.shape[0]))


