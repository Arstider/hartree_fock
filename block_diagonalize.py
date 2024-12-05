import numpy as np

def block_diagonalize(matrices):
    if not isinstance(matrices, list):
        raise TypeError("Input 'matrices' must be a list of NumPy arrays.")
    
    if not all(isinstance(matrix, np.ndarray) for matrix in matrices):
        raise TypeError("All elements in 'matrices' must be NumPy arrays.")
    
    if not matrices:
        raise ValueError("Input 'matrices' cannot be empty.")
    
    total_rows = sum(matrix.shape[0] for matrix in matrices)
    total_cols = sum(matrix.shape[1] for matrix in matrices)

    block_diag_matrix = np.zeros((total_rows, total_cols))

    row_pos = 0
    col_pos = 0

    for matrix in matrices:
        rows, cols = matrix.shape
        block_diag_matrix[row_pos:row_pos+rows, col_pos:col_pos+cols] = matrix
        row_pos += rows
        col_pos += cols

    return block_diag_matrix

