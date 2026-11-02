import sys
import os

# Adjust path so we can import csp_solver if running from root or solvers/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from csp_solver import SudokuCSP

def solve(grid: list[int], seed: int | None = None, **options) -> dict:
    """
    Mandatory solver interface required by benchmark/run_benchmark.py.
    """
    board_2d = [grid[i:i+9] for i in range(0, 81)]
    solver = SudokuCSP(board_2d)
    solved_board_2d = solver.solve()
    
    if solved_board_2d is None:
        return {
            "solution": list(grid),
            "solved": False,
            "steps": solver.revisions
        }
        
    solution_1d = [val for row in solved_board_2d for val in row]
    
    return {
        "solution": solution_1d,
        "solved": True,
        "steps": solver.revisions
    }

def create_sample_sudoku_image(filename="sample_sudoku.png"):
    import cv2
    import numpy as np
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    margin, cell = 25, 450 // 9

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

if __name__ == "__main__":
    print("1. Starting visual pipeline...", flush=True)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    import time
    import cv2
    print("2. Loading TensorFlow...", flush=True)
    import tensorflow as tf
    
    from vision import extract_sudoku_grid, parse_grid_digits
    from display import display_solution

    image_path = "sample_sudoku.png"
    model_path = "digit_model.keras"

    if not os.path.exists(model_path):
        print("Error: Run 'python train_model.py' first.")
        sys.exit(1)

    if not os.path.exists(image_path):
        create_sample_sudoku_image(image_path)

    print("Extracting grid and recognizing digits...")
    model = tf.keras.models.load_model(model_path)
    warped_color, warped_thresh = extract_sudoku_grid(image_path)
    initial_board_2d = parse_grid_digits(warped_thresh, model)
    
    initial_grid_1d = [val for row in initial_board_2d for val in row]

    print("\nSolving using standard team interface...")
    start_time = time.perf_counter()
    
    result = solve(initial_grid_1d)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if not result["solved"]:
        print("No valid solution found. OCR may have misclassified a digit.")
        sys.exit(1)

    print(f"\n[Evaluation Metrics]")
    print(f"Solve Time:    {elapsed_ms:.2f} ms")
    print(f"AC-3 Revisions:{result['steps']}")

    solved_board_2d = [result["solution"][i:i+9] for i in range(0, 81)]
    
    output_img = display_solution(warped_color, initial_board_2d, solved_board_2d, "solved.png")
    print("Saved visual solution to solved.png")
    cv2.imshow("Solved Sudoku", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
