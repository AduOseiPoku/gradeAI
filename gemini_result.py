import cv2
import numpy as np

def align_sheet(image, target_width=1000, target_height=1400):
    """Finds the main sheet boundary and applies perspective warp."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    sheet_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet_contour = approx
            break
            
    if sheet_contour is None:
        # Fallback: if border contour is not isolated, resize directly
        return cv2.resize(image, (target_width, target_height))
    
    # Order points: [top-left, top-right, bottom-right, bottom-left]
    pts = sheet_contour.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (target_width, target_height))

def isolate_pencil_marks(warped_img):
    """Uses the green channel to drop out pink ink and isolate pencil marks."""
    # The green channel suppresses pink/red ink and keeps black marks dark
    green_channel = warped_img[:, :, 1]
    
    # Otsu thresholding inverted (pencil marks become pure white on black background)
    _, thresh = cv2.threshold(green_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def evaluate_sheet(warped_img, binary_mask, answer_key):
    # Normalized coordinate grid for the warped 1000x1400 image
    # Format: (start_x, start_y, spacing_x, spacing_y, bubble_radius)
    COL1_CONFIG = {"start_x": 315, "start_y": 745, "dx": 38, "dy": 29, "r": 10, "start_q": 1, "end_q": 15}
    COL2_CONFIG = {"start_x": 585, "start_y": 745, "dx": 38, "dy": 29, "r": 10, "start_q": 16, "end_q": 30}
    
    options = ['A', 'B', 'C', 'D']
    detected_answers = {}
    score = 0
    annotated_img = warped_img.copy()

    for col in [COL1_CONFIG, COL2_CONFIG]:
        for i, q_num in enumerate(range(col["start_q"], col["end_q"] + 1)):
            q_y = int(col["start_y"] + i * col["dy"])
            bubble_scores = []
            bubble_coords = []
            
            for opt_idx in range(4):
                q_x = int(col["start_x"] + opt_idx * col["dx"])
                r = col["r"]
                bubble_coords.append((q_x, q_y))
                
                # Create mask to count non-zero pixels strictly within the bubble
                mask = np.zeros(binary_mask.shape, dtype="uint8")
                cv2.circle(mask, (q_x, q_y), r, 255, -1)
                
                filled_pixels = cv2.countNonZero(cv2.bitwise_and(binary_mask, binary_mask, mask=mask))
                bubble_scores.append(filled_pixels)
            
            max_val = max(bubble_scores)
            max_idx = bubble_scores.index(max_val)
            sorted_scores = sorted(bubble_scores, reverse=True)
            
            # Shading decision logic
            SHADING_PIXEL_THRESHOLD = 80  # Minimum filled pixels to qualify as marked
            if max_val < SHADING_PIXEL_THRESHOLD:
                chosen = "BLANK"
            elif len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 20:
                chosen = "INVALID"
            else:
                chosen = options[max_idx]
                
            detected_answers[q_num] = chosen
            
            # Grade against answer key and visualize
            correct_opt = answer_key.get(q_num)
            for opt_idx, (bx, by) in enumerate(bubble_coords):
                if options[opt_idx] == correct_opt:
                    cv2.circle(annotated_img, (bx, by), col["r"] + 2, (0, 255, 0), 2)  # Green ring on correct answer
                elif options[opt_idx] == chosen and chosen != correct_opt:
                    cv2.circle(annotated_img, (bx, by), col["r"] + 2, (0, 0, 255), 2)  # Red ring on wrong answer
                    
            if chosen == correct_opt:
                score += 1

    return detected_answers, score, annotated_img

# --- Execution Example ---
if __name__ == "__main__":
    # 1. Define Master Answer Key (Questions 1 to 30)
    ANSWER_KEY = {
        1: 'B',  2: 'D',  3: 'D',  4: 'B',  5: 'C',
        6: 'B',  7: 'C',  8: 'A',  9: 'D', 10: 'B',
        11: 'C', 12: 'D', 13: 'C', 14: 'D', 15: 'B',
        16: 'B', 17: 'C', 18: 'D', 19: 'A', 20: 'A',
        21: 'C', 22: 'D', 23: 'B', 24: 'A', 25: 'C',
        26: 'A', 27: 'C', 28: 'D', 29: 'B', 30: 'A'
    }

    # 2. Load and process
    input_image = cv2.imread(r"C:\Users\adupo\Downloads\scanable sheet.PNG")
    warped = align_sheet(input_image)
    binary_mask = isolate_pencil_marks(warped)
    answers, total_score, result_img = evaluate_sheet(warped, binary_mask, ANSWER_KEY)

    print(f"Total Score: {total_score} / {len(ANSWER_KEY)}")
    print("Detected Answers:", answers)

    cv2.imwrite("graded_sheet_output.png", result_img)