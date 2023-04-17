from .vec4 import *
from .vec3 import *

import numpy as np

class Mat4:
    def __init__(self, identity = False):
        self.m1 = Vec4()
        self.m2 = Vec4()
        self.m3 = Vec4()
        self.m4 = Vec4()

        if identity:
            self.m1.x = 1
            self.m2.y = 1
            self.m3.z = 1
            self.m4.w = 1

    def __str__(self):
        return f"{self.m1}\n{self.m2}\n{self.m3}\n{self.m4}"

    def __getitem__(self, index):
        if index >= 0 and index < 4:
            if index == 0:
                return self.m1
            elif index == 1:
                return self.m2
            elif index == 2:
                return self.m3
            elif index == 3:
                return self.m4
        else:
            raise ValueError(f"Invalid index {index} for Mat4 access. Index must be inside the interval [0, 3]")
    
    def __setitem__(self, index, b):
        if isinstance(b, Vec4):
            if index >= 0 and index < 4:
                if index == 0:
                    self.m1 = b
                elif index == 1:
                    self.m2 = b
                elif index == 2:
                    self.m3 = b
                elif index == 3:
                    self.m4 = b
            else:
                raise ValueError(f"Invalid index {index} for Mat4 access. Index must be inside the interval [0, 3]")
        else:
            raise NotImplementedError
    
    def __add__(self, b):
        if isinstance(b, Mat4):
            add_mat = Mat4()
            add_mat[0] = self.m1 + b
            add_mat[1] = self.m2 + b
            add_mat[2] = self.m3 + b
            add_mat[3] = self.m4 + b
            return add_mat
        else:
            raise NotImplementedError
        return self
    
    def __sub__(self, b):
        if isinstance(b, Mat4):
            sub_mat = Mat4()
            sub_mat[0] = self.m1 - b
            sub_mat[1] = self.m2 - b
            sub_mat[2] = self.m3 - b
            sub_mat[3] = self.m4 - b
            return sub_mat
        else:
            raise NotImplementedError
    
    def __truediv__(self, b):
        if isinstance(b, int) or isinstance(b, float):           
            truediv_mat = Mat4()
            truediv_mat[0] = self.m1 / b
            truediv_mat[1] = self.m2 / b
            truediv_mat[2] = self.m3 / b
            truediv_mat[3] = self.m4 / b
            return truediv_mat
        else:
            raise NotImplementedError
    
    def __mul__(self, b):
        if isinstance(b, Mat4):
            mul_list = np.matmul(self.to_list(), b.to_list())

            mul_mat = Mat4()
            mul_mat.from_list(mul_list)
            return mul_mat
        else:
            raise NotImplementedError
    
    def to_list(self):
        return [self.m1.to_list(), self.m2.to_list(), self.m3.to_list(), self.m4.to_list()]
    
    def from_list(self, b):
        if isinstance(b, list) or isinstance(b, np.ndarray):
            if len(b) == 4:
                self.m1.from_list(b[0])
                self.m2.from_list(b[1])
                self.m3.from_list(b[2])
                self.m4.from_list(b[3])
            else:
                raise ValueError("Invalid number of components from list. Mat4 construction needs a 4 item list")
        else:
            raise NotImplementedError
        
    def transpose(self):
        transpose_list = np.transpose(self.to_list())
        self.from_list(transpose_list)