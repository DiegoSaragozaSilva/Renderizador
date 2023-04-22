from Math.vec2 import *
from Math.vec3 import *
from Math.vec_math import *

from PIL import Image

def bresenham(start: Vec2, end: Vec2) -> list:
    points = list()

    u = start.x
    v = start.y
    du = abs(end.x - start.x)
    dv = abs(end.y - start.y)
    su = 1 if start.x < end.x else -1
    sv = 1 if start.y < end.y else -1
    error = du if du > dv else -dv
    error /= 2
    last_error = 0

    while True:
        points.append(Vec2(u, v))
        
        if (u == end.x and v == end.y):
            break
        last_error = error
        if (last_error > -du):
            error -= dv
            u += su
        if (last_error < dv):
            error += du
            v += sv

    return points

def is_point_inside_triangle(point: Vec2, v0: Vec2, v1: Vec2, v2: Vec2) -> bool:
    Q = point

    P = Vec2(v1.x - v0.x, v1.y - v0.y)
    n = Vec2(P[1], -P[0])
    QLine = Vec2(Q.x - v0.x, Q.y - v0.y)
    dot = QLine.x * n.x + QLine.y * n.y
    if dot < 0:
        return False

    P = Vec2(v2.x - v1.x, v2.y - v1.y)
    n = Vec2(P[1], -P[0])
    QLine = Vec2(Q.x - v1.x, Q.y - v1.y)
    dot = QLine.x * n.x + QLine.y * n.y
    if dot < 0:
        return False

    P = Vec2(v0.x - v2.x, v0.y - v2.y)
    n = Vec2(P[1], -P[0])
    QLine = Vec2(Q.x - v2.x, Q.y - v2.y)
    dot = QLine.x * n.x + QLine.y * n.y
    if dot < 0:
        return False
    
    return True

def homogeneous_divide(mat: np.ndarray) -> np.ndarray:
    mat = np.transpose(mat)
    for row in mat:
        row[0] /= row[3]
        row[1] /= row[3]
        row[2] /= row[3]
        row[3] = 1.0
    return np.transpose(mat)

def get_baricentric_coefficients(v0: Vec2, v1: Vec2, v2: Vec2, point: Vec2) -> Vec3:
    alpha = (-(point.x - v1.x) * (v2.y - v1.y) + (point.y - v1.y) * (v2.x - v1.x)) / (-(v0.x - v1.x) * (v2.y - v1.y) + (v0.y - v1.y) * (v2.x - v1.x))
    beta = (-(point.x - v2.x) * (v0.y - v2.y) + (point.y - v2.y) * (v0.x - v2.x)) / (-(v1.x - v2.x) * (v0.y - v2.y) + (v1.y - v2.y) * (v0.x - v2.x))
    gamma = 1.0 - alpha - beta
    return Vec3(alpha, beta, gamma)

def load_image_buffer(path):
    return Image.open(path).load()

def get_triangles_normals(vertex_list):
    normals = list()
    for i in range(0, len(vertex_list) - 2, 3):
        v0 = Vec3()
        v1 = Vec3()
        v2 = Vec3()

        v0.from_list(vertex_list[i + 0][:3])
        v1.from_list(vertex_list[i + 1][:3])
        v2.from_list(vertex_list[i + 2][:3])

        normal = cross(v2 - v0, v1 - v0)
        normal.normalize()
        normals.append(normal)
    return normals
