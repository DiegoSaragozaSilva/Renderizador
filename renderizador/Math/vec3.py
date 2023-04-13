from math import sqrt
import numpy as np

class Vec3:
    x = 0
    y = 0
    z = 0
    
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"[{self.x:.4f}, {self.y:.4f}, {self.z:.4f}]"
    
    def __getitem__(self, index):
        if index >= 0 and index < 3:
            if index == 0:
                return self.x
            elif index == 1:
                return self.y
            elif index == 2:
                return self.z
        raise ValueError("Invalid index for Vec3 access. Index must be inside the interval [0, 2]")
    
    def __setitem__(self, index, b):
        if isinstance(b, int) or isinstance(b, float):
            if index >= 0 and index < 3:
                if index == 0:
                    self.x = b
                elif index == 1:
                    self.y = b
                elif index == 2:
                    self.z = b
            else:
                raise ValueError("Invalid index for Vec3 access. Index must be inside the interval [0, 2]")
        else:
            raise NotImplementedError

    def __add__(self, b):
        if isinstance(b, Vec3):
            self.x += b.x
            self.y += b.y
            self.z += b.z
        elif isinstance(b, int) or isinstance(b, float):
            self.x += b
            self.y += b
            self.z += b
        else:
            raise NotImplementedError
        return self

    def __sub__(self, b):
        if isinstance(b, Vec3):
            self.x -= b.x
            self.y -= b.y
            self.z -= b.z
        elif isinstance(b, int) or isinstance(b, float):
            self.x -= b
            self.y -= b
            self.z -= b
        else:
            raise NotImplementedError
        return self

    def __mul__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            self.x *= b
            self.y *= b
            self.z *= b
        else:
            raise NotImplementedError
        return self
    
    def __truediv__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            self.x /= b
            self.y /= b
            self.z /= b
        else:
            raise NotImplementedError
        return self
    
    def length(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self):
        self /= self.length()

    def to_list(self):
        return [self.x, self.y, self.z]
    
    def from_list(self, b):
        if isinstance(b, list) or isinstance(b, np.ndarray):
            if len(b) == 3:
                self.x = b[0]
                self.y = b[1]
                self.z = b[2]
            else:
                raise ValueError("Invalid number of components from list. Vec3 construction needs a 3 item list")
        else:
            raise NotImplementedError