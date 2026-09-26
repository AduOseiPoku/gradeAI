import cv2
import numpy as np

def find_corner_markers(image_path):
    # 1. Load image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Thresholding
    # Use Otsu's thresholding to automatically find the best binarization threshold
    # INVERT it so black shapes (our corner markers) become white (255)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corner_markers = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # We are looking for roughly square shapes (4 corners)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust these values based on actual image resolution)
            if 500 < area < 50000:
                # Get the bounding box center to represent the corner marker point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    corner_markers.append((cX, cY))
                    
    # We need exactly 4 corner markers
    if len(corner_markers) != 4:
        raise ValueError(f"Found {len(corner_markers)} corner markers instead of 4. Tuning required.")
        
    return img, corner_markers

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def warp_image(img, corner_markers, width=800, height=1000):
    rect = order_points(corner_markers)
    
    # Construct the destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
        
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    return warped

# Configuration: Fixed coordinates based on a warped 800x1000 image
# You will manually measure these (X, Y) pixel coordinates from your first perfect warped image.
# For simplicity, here is an example mapping of the first bubble (A) for two questions, 
# and the horizontal pixel spacing between options (A to B).

START_X_Q1 = 150
START_Y_Q1 = 300
OPTION_SPACING_X = 40 # Pixels between A, B, C, D, E bubbles

QUESTIONS_CONFIG = {
    1: {"start_x": START_X_Q1, "start_y": START_Y_Q1, "options": 4},
    2: {"start_x": START_X_Q1, "start_y": START_Y_Q1 + 50, "options": 5}, # 50px vertical spacing between questions
    # ... mapped up to 20
}

def get_fill_ratio(warped_img, x, y, radius=12):
    # Crop the Region of Interest (ROI) around the bubble center
    roi = warped_img[y-radius:y+radius, x-radius:x+radius]
    
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate how dark it is (0 = white, 255 = black)
    # np.mean gets the average pixel intensity
    fill_ratio = 1 - (np.mean(gray_roi) / 255.0) 
    return fill_ratio

def extract_answers(warped_img):
    results = {}
    options_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    
    # Threshold to determine if a bubble is actually shaded (needs tuning)
    SHADED_THRESHOLD = 0.4 
    
    for q_num, config in QUESTIONS_CONFIG.items():
        start_x = config["start_x"]
        start_y = config["start_y"]
        num_options = config["options"]
        
        bubble_ratios = []
        
        # Sample each option (A, B, C, D, and optionally E)
        for opt_idx in range(num_options):
            bx = start_x + (opt_idx * OPTION_SPACING_X)
            by = start_y
            
            ratio = get_fill_ratio(warped_img, bx, by)
            bubble_ratios.append(ratio)
            
            # Optional: Draw the sampled ROI for visual debugging
            cv2.circle(warped_img, (bx, by), 12, (0, 0, 255), 2)
            
        # Find the darkest bubble
        max_ratio = max(bubble_ratios)
        max_idx = bubble_ratios.index(max_ratio)
        
        # Sort to check the difference between the darkest and second-darkest bubble
        sorted_ratios = sorted(bubble_ratios, reverse=True)
        
        if max_ratio < SHADED_THRESHOLD:
            # Nothing was shaded dark enough
            results[q_num] = "BLANK"
        elif len(sorted_ratios) > 1 and (sorted_ratios[0] - sorted_ratios[1]) < 0.1:
            # The top two bubbles are too close in darkness (double shaded / poorly erased)
            results[q_num] = "AMBIGUOUS"
        else:
            # We have a clear winner
            results[q_num] = options_map[max_idx]
            
    return results, warped_img

def main():
        image_path = r"C:\Users\adupo\Downloads\scanable sheet.PNG"
    
        try:
            # Step 1
            img, corner_markers = find_corner_markers(image_path)
            
            # Step 2
            warped = warp_image(img, corner_markers)
            
            # Step 3
            answers, debug_img = extract_answers(warped)
            
            print("Extracted Answers:")
            for q, a in answers.items():
                print(f"Q{q}: {a}")
                
            # Display the warped image with our sampling circles drawn on it
            cv2.imshow("Debug Output", debug_img)
            cv2.waitKey(0)
        
        except Exception as e:
            print(f"Error processing sheet: {e}")

if __name__ == "__main__":
    main()
