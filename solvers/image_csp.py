import os
import time
import cv2
import numpy as np
import tensorflow as tf

from vision import extract_sudoku_grid, parse_grid_digits
from csp_solver import SudokuCSP
from display import display_solution

def create_sample_sudoku_image(filename="sample_sudoku.png"):
    """Generates a clean test Sudoku image so you can test immediately."""
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    margin = 25
    cell = 450 // 9

    for i in range(10):
        thickness = 4 if i % 3 == 0 else 1
        pos = margin + i * cell
        cv2.line(img, (margin, pos), (margin + 450, pos), (0, 0, 0), thickness)
        cv2.line(img, (pos, margin), (pos, margin + 450), (0, 0, 0), thickness)

    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                x = margin + c * cell + 15
                y = margin + (r + 1) * cell - 13
                cv2.putText(
                    img, str(board[r][c]), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                )

    cv2.imwrite(filename, img)
    print(f"Created test puzzle image: {filename}")

def print_board(board, title):
    print(f"\n--- {title} ---")
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row_str = " ".join(
            str(board[r][c]) if board[r][c] != 0 else "." for c in range(9)
        )
        print(f"{row_str[:5]} | {row_str[6:11]} | {row_str[12:]}")

def main():
    image_path = "sample_sudoku.png"
    model_path = "digit_model.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Run 'python train_model.py' first to create digit_model.keras.")

    if not os.path.exists(image_path):
        create_sample_sudoku_image(image_path)

    # 1. Module (a): Image Recognition
    print("Loading CNN model and extracting grid...")
    model = tf.keras.models.load_model(model_path)
    warped_color, warped_thresh = extract_sudoku_grid(image_path)
    initial_board = parse_grid_digits(warped_thresh, model)
    print_board(initial_board, "Recognized Board from Image")

    # 2. Module (b): Solve with CSP + AC-3
    print("\nSolving puzzle using CSP (AC-3 + MAC)...")
    start_time = time.perf_counter()
    solver = SudokuCSP(initial_board)
    solved_board = solver.solve()
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if solved_board is None:
        print("No valid solution found. Check if OCR misclassified a digit.")
        return

    print_board(solved_board, "Solved Sudoku Board")
    print(f"\n[Evaluation Metrics]")
    print(f"Solve Time:    {elapsed_ms:.2f} ms")
    print(f"AC-3 Revisions: {solver.revisions}")

    # 3. Module (c): Display Solution
    output_img = display_solution(warped_color, initial_board, solved_board, "solved.png")
    print("Saved visual solution to solved.png")

    cv2.imshow("Solved Sudoku", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()