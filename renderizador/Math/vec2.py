from math import sqrt
import numpy as np

class Vec2:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"[{self.x}, {self.y}]"
    
    def __getitem__(self, index):
        if index >= 0 and index < 2:
            if index == 0:
                return self.x
            elif index == 1:
                return self.y
        raise ValueError("Invalid index for Vec2 access. Index must be inside the interval [0, 1]")
    
    def __setitem__(self, index, b):
        if isinstance(b, int) or isinstance(b, float):
            if index >= 0 and index < 2:
                if index == 0:
                    self.x = b
                elif index == 1:
                    self.y = b
            else:
                raise ValueError("Invalid index for Vec2 access. Index must be inside the interval [0, 1]")
        else:
            raise NotImplementedError

    def __add__(self, b):
        if isinstance(b, Vec2):
            return Vec2(self.x + b.x, self.y + b.y)
        elif isinstance(b, int) or isinstance(b, float):
            return Vec2(self.x + b, self.y + b)
        else:
            raise NotImplementedError

    def __sub__(self, b):
        if isinstance(b, Vec2):
            return Vec2(self.x - b.x, self.y - b.y)
        elif isinstance(b, int) or isinstance(b, float):
            return Vec2(self.x - b, self.y - b)
        else:
            raise NotImplementedError

    def __mul__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            return Vec2(self.x * b, self.y * b,)
        else:
            raise NotImplementedError
    
    def __truediv__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            return Vec2(self.x / b, self.y / b)
        else:
            raise NotImplementedError
    
    def length(self):
        return sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self):
        length = self.length()
        self.x /= length
        self.y /= length
        return self

    def to_list(self):
        return [self.x, self.y]
    
    def from_list(self, b):
        if isinstance(b, list) or isinstance(b, np.ndarray):
            if len(b) == 2:
                self.x = b[0]
                self.y = b[1]
            else:
                raise ValueError("Invalid number of components from list. Vec2 construction needs a 2 item list")
        else:
            raise NotImplementedError
