from vec3 import *

class AABB:
    min = Vec3()
    max = Vec3()

    def __init__(self, min: Vec3 = Vec3(), max: Vec3 = Vec3()):
        self.min = min
        self.max = max

    def create_from_triangle(self, v0: Vec3, v1: Vec3, v2: Vec3):
        self.min = Vec3(int(np.floor(min(v0.x, v1.x, v2.x))), int(np.floor(min(v0.y, v1.y, v2.y))), int(np.floor(min(v0.z, v1.z, v2.z))))
        self.max = Vec3(int(np.ceil(max(v0.x, v1.x, v2.x))), int(np.ceil(max(v0.y, v1.y, v2.y))), int(np.ceil(max(v0.z, v1.z, v2.z))))
                    