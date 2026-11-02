import cv2

def display_solution(warped_color, initial_board, solved_board, output_path="solved.png"):
    output_img = warped_color.copy()
    cell_size = 450 // 9

    for r in range(9):
        for c in range(9):
            if initial_board[r][c] == 0 and solved_board[r][c] != 0:
                text = str(solved_board[r][c])
                x = c * cell_size + 15
                y = (r + 1) * cell_size - 12
                cv2.putText(
                    output_img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 0), 2, cv2.LINE_AA
                )

    cv2.imwrite(output_path, output_img)
    return output_img
    
