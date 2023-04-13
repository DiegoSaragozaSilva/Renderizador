from vec4 import *
from vec3 import *
import numpy as np

class Mat4:
    m1 = Vec4()
    m2 = Vec4()
    m3 = Vec4()
    m4 = Vec4()

    def __init__(self, identity = False):
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
            self.m1 += b.m1
            self.m2 += b.m2
            self.m3 += b.m3
            self.m4 += b.m4
        else:
            raise NotImplementedError
        return self
    
    def __sub__(self, b):
        if isinstance(b, Mat4):
            self.m1 -= b.m1
            self.m2 -= b.m2
            self.m3 -= b.m3
            self.m4 -= b.m4
        else:
            raise NotImplementedError
        return self
    
    def __truediv__(self, b):
        if isinstance(b, int) or isinstance(b, float):
            self.m1 /= b
            self.m2 /= b
            self.m3 /= b
            self.m4 /= b
        else:
            raise NotImplementedError
        return self
    
    def __mul__(self, b):
        if isinstance(b, Mat4):
            mul_list = np.matmul(self.to_list(), b.to_list())
            self.from_list(mul_list)
        else:
            raise NotImplementedError
        return self
    
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
        return self

    def homogeneous_divide(self):
        self.transpose()
        self_list = self.to_list()
        for row in self_list:
            row[0] /= row[3]
            row[1] /= row[3]
            row[2] /= row[3]
            row[3] /= 1
        self.from_list(self_list)
        self.transpose()
        
    def translate(self, translation: Vec3):
        translation_matrix = Mat4(identity = True)
        translation_matrix[0] = Vec4(1, 0, 0, translation.x)
        translation_matrix[1] = Vec4(0, 1, 0, translation.y)
        translation_matrix[2] = Vec4(0, 0, 1, translation.z)
        translation_matrix[3] = Vec4(0, 0, 0, 1)

        self = translation_matrix * self
        return self
    
    def scale(self, scale: Vec3):
        scale_matrix = Mat4(identity = True)
        scale_matrix[0] = Vec4(scale[0], 0, 0, 0)
        scale_matrix[1] = Vec4(0, scale[1], 0, 0)
        scale_matrix[2] = Vec4(0, 0, scale[2], 0)
        scale_matrix[3] = Vec4(0, 0, 0, 1)

        self = scale_matrix * self
        return self

    def rotate(self, angle: float, axis: Vec3):
        quaternion = Vec4()
        quaternion.x = axis.x * np.sin(angle / 2.0)
        quaternion.y = axis.y * np.sin(angle / 2.0)
        quaternion.z = axis.z * np.sin(angle / 2.0)
        quaternion.w = np.cos(angle / 2.0)

        rotation_matrix = Mat4(identity = True)
        rotation_matrix[0] = Vec4(1 - 2 * (quaternion.y**2 + quaternion.z**2), 2 * (quaternion.x * quaternion.y - quaternion.z * quaternion.w), 2 * (quaternion.x * quaternion.z + quaternion.y * quaternion.w), 0)
        rotation_matrix[1] = Vec4(2 * (quaternion.x * quaternion.y + quaternion.z * quaternion.w), 1 - 2 * (quaternion.x**2 + quaternion.z**2), 2 * (quaternion.y * quaternion.z + quaternion.x * quaternion.w), 0)
        rotation_matrix[2] = Vec4(2 * (quaternion.x * quaternion.z - quaternion.y * quaternion.w), 2 * (quaternion.y * quaternion.z + quaternion.x * quaternion.w), 1 - 2 * (quaternion.x**2 + quaternion.y**2), 0)
        rotation_matrix[3] = Vec4(0, 0, 0, 1)

        self = rotation_matrix * self
        return self

    def look_at(self, eye: Vec3, orientation: Vec4):
        rotation_angle = orientation.w
        rotation_axis = Vec3(orientation.x, orientation.y, orientation.z)

        rotation_matrix = Mat4(identity = True)
        rotation_matrix = rotation_matrix.rotate(rotation_angle, rotation_axis)
        rotation_matrix = rotation_matrix.transpose()

        translation_matrix = Mat4(identity = True)
        translation_matrix = translation_matrix.translate(eye * -1)

        self = rotation_matrix * translation_matrix
        print("lookat")
        print(self, "\n")
        return self

    def perspective(self, fovy: float, aspect_ratio: float, near: float, far: float):
        top = near * np.tan(fovy)
        right = top * aspect_ratio

        projection_matrix = Mat4(identity = True)
        projection_matrix[0] = Vec4(near / right, 0, 0, 0)
        projection_matrix[1] = Vec4(0, near / top, 0, 0)
        projection_matrix[2] = Vec4(0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near))
        projection_matrix[3] = Vec4(0, 0, -1, 0)
        print("persp")
        print(projection_matrix, "\n")

        self = projection_matrix * self

        return self