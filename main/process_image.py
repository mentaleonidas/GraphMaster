import cv2
import numpy as np
import os
from pathlib import Path
import json
from text_Rekognition import detectText
from axes_detection import detectAxes
from color_cluster import color_cluster
from Data_Extraction import data_of_one_line

def process_image(image_path, debug=False):
    """
    Process a single image to extract graph data
    Returns JSON with extracted data points and metadata
    """
    # Initialize AWS client
    client = boto3.client('rekognition', region_name='us-west-2')
    
    # Initialize text detection dictionaries
    img_text = {}
    image_text = {}
    
    # Read and process image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect axes
    xaxis, yaxis, x_up_axis, y_right_axis = detectAxes(image_path)
    
    if xaxis is None or yaxis is None:
        return {
            "status": "error",
            "message": "Could not detect axes in the image"
        }
    
    xaxis_yvalue = xaxis[1]-1
    yaxis_xvalue = yaxis[0]+1
    up_yvalue = x_up_axis[1]+1
    right_xvalue = y_right_axis[0]-1
    
    # Detect text
    image = detectText(Path(image_path).name, image, image_text, img_text)
    
    # Crop image
    image_crop_del = image[x_up_axis[1]+1:xaxis[1], yaxis[0]+1:y_right_axis[0]]
    
    # Extract coordinates
    x_coords = []
    y_coords = []
    
    for i in range(len(img_text[Path(image_path).name])):
        if img_text[Path(image_path).name][i][0].replace('.','',1).isdigit():
            if img_text[Path(image_path).name][i][1][0] + img_text[Path(image_path).name][i][1][2] < yaxis_xvalue:
                axis_text = img_text[Path(image_path).name][i][0]
                bbox = img_text[Path(image_path).name][i][1]
                y_mid = bbox[1] + bbox[3]/2
                y_coords.append([float(axis_text), bbox, y_mid])
                
            if img_text[Path(image_path).name][i][1][1] > xaxis_yvalue:
                axis_text = img_text[Path(image_path).name][i][0]
                bbox = img_text[Path(image_path).name][i][1]
                x_mid = bbox[0] + bbox[2]/2
                x_coords.append([int(axis_text), bbox, x_mid])
    
    # Color clustering
    image_crop_del = cv2.cvtColor(image_crop_del, cv2.COLOR_RGB2BGR)
    line_imgs = color_cluster(image_crop_del, debug=debug)
    
    # Process each line
    extracted_data = []
    for i, line_img in enumerate(line_imgs):
        line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2BGR)
        data_points = data_of_one_line(line_img, debug=debug)
        
        if data_points:
            # Convert pixel coordinates to actual values
            data_points = np.array(data_points)
            data_points[:,0] += yaxis_xvalue
            data_points[:,1] += up_yvalue
            
            # Sort by x values
            sort_index = np.lexsort((data_points[:,1], data_points[:,0]))
            data_points = data_points[sort_index,:]
            
            # Convert to actual values using axis coordinates
            converted_points = []
            for point in data_points:
                x_val = convert_pixel_to_value(point[0], x_coords)
                y_val = convert_pixel_to_value(point[1], y_coords)
                if x_val is not None and y_val is not None:
                    converted_points.append([x_val, y_val])
            
            if converted_points:
                extracted_data.append({
                    "line_id": i,
                    "points": converted_points
                })
    
    return {
        "status": "success",
        "data": {
            "axes": {
                "x": x_coords,
                "y": y_coords
            },
            "extracted_lines": extracted_data
        }
    }

def convert_pixel_to_value(pixel, coords):
    """
    Convert pixel coordinate to actual value using axis coordinates
    """
    if not coords:
        return None
        
    coords = np.array(coords)
    pixel_coords = coords[:, 2]  # Get the pixel coordinates
    values = coords[:, 0]  # Get the actual values
    
    # Find the closest coordinates
    idx = np.searchsorted(pixel_coords, pixel)
    
    if idx == 0:
        return values[0]
    if idx == len(coords):
        return values[-1]
        
    # Interpolate between the two closest points
    x1, x2 = pixel_coords[idx-1], pixel_coords[idx]
    y1, y2 = values[idx-1], values[idx]
    
    return y1 + (pixel - x1) * (y2 - y1) / (x2 - x1) 