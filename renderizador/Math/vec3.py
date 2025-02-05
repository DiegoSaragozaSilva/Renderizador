from math import sqrt
import numpy as np

class Vec3:    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"[{self.x}, {self.y}, {self.z}]"
    
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
            return Vec3(self.x + b.x, self.y + b.y, self.z + b.z)
        elif isinstance(b, int) or isinstance(b, float):
            return Vec3(self.x + b, self.y + b, self.z + b)
        else:
            raise NotImplementedError

    def __sub__(self, b):
        if isinstance(b, Vec3):
            return Vec3(self.x - b.x, self.y - b.y, self.z - b.z)
        elif isinstance(b, int) or isinstance(b, float):
            return Vec3(self.x - b, self.y - b, self.z - b)
        else:
            raise NotImplementedError

    def __mul__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            return Vec3(self.x * b, self.y * b, self.z * b)
        elif isinstance(b, Vec3):
            return Vec3(self.x * b.x, self.y * b.y, self.z * b.z)
        else:
            raise NotImplementedError
    
    def __truediv__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            return Vec3(self.x / b, self.y / b, self.z / b)
        else:
            raise NotImplementedError
    
    def length(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self):
        length = self.length()
        if length == 0.0:
            return

        self.x /= length
        self.y /= length
        self.z /= length
        return self

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
