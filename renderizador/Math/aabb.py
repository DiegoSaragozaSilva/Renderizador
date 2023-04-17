from .vec2 import *

class AABB:
    min = Vec2()
    max = Vec2()

    def __init__(self, min: Vec2 = Vec2(), max: Vec2 = Vec2()):
        self.min = min
        self.max = max

    def create_from_triangle(self, v0: Vec2, v1: Vec2, v2: Vec2):
        self.min = Vec2(np.floor(min(v0.x, v1.x, v2.x)), np.floor(min(v0.y, v1.y, v2.y)))
        self.max = Vec2(np.ceil(max(v0.x, v1.x, v2.x)), np.ceil(max(v0.y, v1.y, v2.y)))

    def intersects(self, b) -> bool:
        return min(self.max.x, b.max.x) > max(self.min.x, b.min.x) and min(self.max.y, b.max.y) > max(self.min.y, b.min.y)
    
    def intersection_square(self, b):
        return AABB(Vec2(max(self.min.x, b.min.x), max(self.min.y, b.min.y)),
                    Vec2(min(self.max.x, b.max.x), min(self.max.y, b.max.y)))

