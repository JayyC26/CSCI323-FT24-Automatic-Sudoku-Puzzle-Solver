import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left
    return rect

def extract_sudoku_grid(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_Cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzle_Cnt = approx.reshape(4, 2)
            break

    if puzzle_Cnt is None:
        raise ValueError("Could not find a 4-corner Sudoku grid contour.")

    rect = order_points(puzzle_Cnt)
    dst = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_color = cv2.warpPerspective(img, M, (450, 450))
    warped_thresh = cv2.warpPerspective(thresh, M, (450, 450))

    return warped_color, warped_thresh

def parse_grid_digits(warped_thresh, model):
    board = np.zeros((9, 9), dtype=int)
    cell_size = 450 // 9

    for r in range(9):
        for c in range(9):
            cell = warped_thresh[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size]
            # Crop 6px border to remove grid lines
            cropped = cell[6:-6, 6:-6]
            
            # Pixel count feature from the paper: skip empty cells
            if cv2.countNonZero(cropped) < 40:
                board[r, c] = 0
                continue

            resized = cv2.resize(cropped, (28, 28))
            normalized = resized.astype("float32") / 255.0
            reshaped = np.expand_dims(normalized, axis=(0, -1))

            probs = model.predict(reshaped, verbose=0)[0]
            # Predict 1-9 (exclude class 0 if using a 10-class model)
            board[r, c] = int(np.argmax(probs[1:]) + 1)

    return board