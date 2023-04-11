from param import Parameter as p
import numpy as np

########
#設計変数の行列(M×N)をベクトル(1×MN)に変換する関数
########
def matrix_to_vector(trajectory_matrix):
    
    trajectory_vector = trajectory_matrix.flatten()
    
    return trajectory_vector

########
#設計変数のベクトル(1×MN)を行列(M×N)をに変換する関数
########
def vector_to_matrix(trajectory_vector):
    
    trajectory_matrix = trajectory_vector.reshape(p.M, p.N)
    
    return trajectory_matrix


    