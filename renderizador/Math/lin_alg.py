from .vec3 import Vec3
from .vec4 import Vec4
from .mat4 import Mat4

import numpy as np

def translate(mat: Mat4, translation: Vec3) -> Mat4:
    translation_matrix = Mat4(identity = True)
    translation_matrix[0] = Vec4(1, 0, 0, translation.x)
    translation_matrix[1] = Vec4(0, 1, 0, translation.y)
    translation_matrix[2] = Vec4(0, 0, 1, translation.z)
    translation_matrix[3] = Vec4(0, 0, 0, 1)

    return translation_matrix * mat

def scale(mat: Mat4, scale: Vec3) -> Mat4:
    scale_matrix = Mat4(identity = True)
    scale_matrix[0] = Vec4(scale[0], 0, 0, 0)
    scale_matrix[1] = Vec4(0, scale[1], 0, 0)
    scale_matrix[2] = Vec4(0, 0, scale[2], 0)
    scale_matrix[3] = Vec4(0, 0, 0, 1)

    return scale_matrix * mat

def rotate(mat: Mat4, angle: float, axis: Vec3) -> Mat4:
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

    return rotation_matrix * mat

def look_at(eye: Vec3, orientation: Vec4) -> Mat4:
    rotation_angle = orientation.w
    rotation_axis = Vec3(orientation.x, orientation.y, orientation.z)

    rotation_matrix = Mat4(identity = True)
    rotation_matrix = rotate(rotation_matrix, rotation_angle, rotation_axis)
    rotation_matrix.transpose()

    translation_matrix = Mat4(identity = True)
    translation_matrix = translate(translation_matrix, eye * -1)

    return rotation_matrix * translation_matrix

def perspective(fovy: float, aspect_ratio: float, near: float, far: float) -> Mat4:
    top = near * np.tan(fovy)
    right = top * aspect_ratio

    projection_matrix = Mat4(identity = True)
    projection_matrix[0] = Vec4(near / right, 0, 0, 0)
    projection_matrix[1] = Vec4(0, near / top, 0, 0)
    projection_matrix[2] = Vec4(0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near))
    projection_matrix[3] = Vec4(0, 0, -1, 0)

    return projection_matrix