
import numpy as np
from numpy import array

dataset=[1,2,3,4,5,6,7,8,9,10]

def split_sequence(sequence, n_steps):
    X, y =list(), list()
    for i in range(len(sequence)):
        end_ix=i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)        
    return array(X),array(y)

split_sequence(dataset,3)
    

