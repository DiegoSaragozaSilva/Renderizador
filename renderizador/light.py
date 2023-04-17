from Math.vec3 import Vec3

class Light:

    def __init__(self, direction: Vec3 = Vec3(), color: Vec3 = Vec3(), intensity: float = 0.0, is_directional: bool = False, is_spot: bool = False, is_point: bool = False):
        self.is_directional = is_directional
        self.is_spot = is_spot
        self.is_point = is_point

        self.direction = direction
        self.color = color
        self.intensity = intensity
