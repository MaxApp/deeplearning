import numpy as np
import pandas as pd

# This is kind of dynamic programming
def min_edit_distance(source, target, insert_cost=1, del_cost=1, replace_cost=2):
    """
    Given a source string and operate by 3 methods of `insert`, `delete`, `replace` 
    with respectively cost, caculate the minimum edit distance required to convert
    the source to target.

    return: the matrix of minimum edit distances
    """

    num_rows = len(source) + 1
    num_cols = len(target) + 1

    matrix = np.zeros((num_rows, num_cols), dtype=int) 

    # initial first row with `insert` step
    for col in range(1, num_cols):
        matrix[0][col] = matrix[0][col-1] + insert_cost

    # inital first col with `insert` step
    for row in range(1, num_rows):
        matrix[row][0] = matrix[0][row-1] + del_cost

    # calculate diagonal with `replace` step
    # print(f"shape: {matrix.shape},  num_cols: {num_cols}")
    for row in range(1, num_rows):
        for col in range(1, num_cols):
            rep_cost = replace_cost
            # print(f"col: {col}")
            if source[row-1] == target[col-1]:
                rep_cost = 0
            matrix[row][col] = min(matrix[row-1][col]+del_cost, matrix[row][col-1]+insert_cost, matrix[row-1][col-1]+rep_cost)
        
    return matrix

if __name__ == "__main__":
    source = "precede"
    target = "proceed"
    m = min_edit_distance(source, target)
    idx = list('#' + source)
    cols = list('#' + target)
    df = pd.DataFrame(m, index=idx, columns= cols)
    print(df)




