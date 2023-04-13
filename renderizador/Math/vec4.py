from math import sqrt
import numpy as np

class Vec4:
    x = 0
    y = 0
    z = 0
    w = 0
    
    def __init__(self, x = 0, y = 0, z = 0, w = 0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __str__(self):
        return f"[{self.x:.4f}, {self.y:.4f}, {self.z:.4f}, {self.w:.4f}]"
    
    def __getitem__(self, index):
        if index >= 0 and index < 4:
            if index == 0:
                return self.x
            elif index == 1:
                return self.y
            elif index == 2:
                return self.z
            elif index == 3:
                return self.w
        raise ValueError("Invalid index for Vec4 access. Index must be inside the interval [0, 3]")
    
    def __setitem__(self, index, b):
        if isinstance(b, int) or isinstance(b, float):
            if index >= 0 and index < 4:
                if index == 0:
                    self.x = b
                elif index == 1:
                    self.y = b
                elif index == 2:
                    self.z = b
                elif index == 3:
                    self.w = b
            else:
                raise ValueError("Invalid index for Vec4 access. Index must be inside the interval [0, 3]")
        else:
            raise NotImplementedError

    def __add__(self, b):
        if isinstance(b, Vec4):
            self.x += b.x
            self.y += b.y
            self.z += b.z
            self.w += b.w
        elif isinstance(b, int) or isinstance(b, float):
            self.x += b
            self.y += b
            self.z += b
            self.w += b
        else:
            raise NotImplementedError
        return self

    def __sub__(self, b):
        if isinstance(b, Vec4):
            self.x -= b.x
            self.y -= b.y
            self.z -= b.z
            self.w -= b.w
        elif isinstance(b, int) or isinstance(b, float):
            self.x -= b
            self.y -= b
            self.z -= b
            self.w -= b
        else:
            raise NotImplementedError
        return self

    def __mul__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            self.x *= b
            self.y *= b
            self.z *= b
            self.w *= b
        else:
            raise NotImplementedError
        return self
    
    def __truediv__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            self.x /= b
            self.y /= b
            self.z /= b
            self.w /= b
        else:
            raise NotImplementedError
        return self
    
    def length(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
    
    def normalize(self):
        self /= self.length()

    def to_list(self):
        return [self.x, self.y, self.z, self.w]
    
    def from_list(self, b):
        if isinstance(b, list) or isinstance(b, np.ndarray):
            if len(b) == 4:
                self.x = b[0]
                self.y = b[1]
                self.z = b[2]
                self.w = b[3]
            else:
                raise ValueError("Invalid number of components from list. Vec4 construction needs a 4 item list")
        else:
            raise NotImplementedError