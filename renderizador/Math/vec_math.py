from .vec2 import *
from .vec3 import *

def dot(a, b):
    if isinstance(a, Vec2) and isinstance(b, Vec2):
        return a.x * b.x + a.y * b.y
    elif isinstance(a, Vec3) and isinstance(b, Vec3):
        return a.x * b.x + a.y * b.y + a.z * b.z
    else:
        raise ValueError("Invalid vector types. Both arguments must be Vec2 or Vec3")
    
def cross(a, b):
    if isinstance(a, Vec2) and isinstance(b, Vec2):
        return Vec3(0, 0, a.x * b.y - a.y * b.x)
    elif isinstance(a, Vec3) and isinstance(b, Vec3):
        return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
    else:
        raise ValueError("Invalid vector types. Both arguments must be Vec2 or Vec3")
