#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Diego Saragoza da Silva 
Disciplina: Computação Gráfica
Data: 10/02/2023
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

import utils
from stack import Stack

from Math.vec2 import *
from Math.vec3 import *
from Math.mat4 import *
from Math.aabb import *

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    
    transform_stack = Stack()
    transform_stack.push(Mat4(identity = True))

    viewpoint_matrix = Mat4(identity = True)

    draw_buffer = None
    depth_buffer = None

    @staticmethod
    def setup(width, height, draw_buffer, depth_buffer, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.draw_buffer = draw_buffer
        GL.depth_buffer = depth_buffer

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""

        emissive_color = [int(255 * c) for c in colors["emissiveColor"]]
        for i in range(0, len(point), 2):
            uv = Vec2(int(point[i + 0]), int(point[i + 1]))
            gpu.GPU.draw_pixel(uv.to_list(), gpu.GPU.RGB8, emissive_color)

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""

        emissive_color = [int(255 * c) for c in colors["emissiveColor"]]
        for i in range(0, len(lineSegments) - 2, 2):
            start_point = Vec2(int(lineSegments[i + 0]), int(lineSegments[i + 1]))
            end_point   = Vec2(int(lineSegments[i + 2]), int(lineSegments[i + 3]))
            pixel_points = utils.bresenham(start_point, end_point)

            for point in pixel_points:
                if (point.x >= 0 and point.y >= 0 and point.x < GL.width and point.y < GL.height):
                    gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.RGB8, emissive_color)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D

        emissive_color = [int(255 * c) for c in colors["emissiveColor"]]
        for i in range(0, len(vertices), 6):
            v0 = Vec2(vertices[i + 0], vertices[i + 1])
            v1 = Vec2(vertices[i + 2], vertices[i + 3])
            v2 = Vec2(vertices[i + 4], vertices[i + 5])

            # Find the triangle AABB
            triangle_aabb = AABB()
            triangle_aabb.create_from_triangle(Vec3(v0.x, v0.y, 0.0), Vec3(v1.x, v1.y, 0.0), Vec3(v2.x, v2.y, 0.0))
            
            # Fill the triangle
            for v in range(triangle_aabb.min.y, triangle_aabb.max.y):
                for u in range(triangle_aabb.min.x, triangle_aabb.max.x):
                    point = Vec2(u, v)
                    if (point.x < 0 or point.x >= GL.width) or (point.y < 0 or point.y >= GL.height) or not utils.is_point_inside_triangle(point, v0, v1, v2):
                        continue
                    gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.RGB8, emissive_color)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        
        # Tive que usar Numpy por que não tive paciência de fazer uma classe de matriz genérica. Talvez eu faça depois...
        points_matrix = list()
        for i in range(0, len(point), 3):
            points_matrix.append([point[i + 0], point[i + 1], point[i + 2], 1])
        points_matrix = np.transpose(np.array(points_matrix))
        
        model_matrix = GL.transform_stack.peek()
        transformed_points = np.matmul(model_matrix.to_list(), points_matrix)
        print("model")
        print(model_matrix, "\n")

        viewpoint_matrix = GL.viewpoint_matrix
        projected_points = np.matmul(viewpoint_matrix.to_list(), transformed_points)
        print("view")
        print(viewpoint_matrix, "\n")

        projected_points = utils.homogeneous_divide(projected_points)
        projected_points = np.transpose(projected_points)

        # Rasterize the triangles
        point_list = list()
        z_list = list()

        for point in projected_points:
            point_list.append(Vec2(point[0], point[1]))
            z_list.append(point[3])

        color_per_vertex = True if isinstance(colors, list) else False

        for i in range(0, len(point_list), 3):
            v0 = point_list[i + 0]
            v1 = point_list[i + 1]
            v2 = point_list[i + 2]
            z0 = z_list[i + 0]
            z1 = z_list[i + 1]
            z2 = z_list[i + 2]

            c0 = Vec3()
            c1 = Vec3()
            c2 = Vec3()
            if color_per_vertex:  
                c0.from_list(colors[(i // 3 * 9) + 0 : (i // 3 * 9) + 3])
                c1.from_list(colors[(i // 3 * 9) + 3 : (i // 3 * 9) + 6])
                c2.from_list(colors[(i // 3 * 9) + 6 : (i // 3 * 9) + 9])

            triangle_aabb = AABB()
            triangle_aabb.create_from_triangle(Vec3(v0.x, v0.y, 0.0), Vec3(v1.x, v1.y, 0.0), Vec3(v2.x, v2.y, 0.0))
            for v in range(triangle_aabb.min.y, triangle_aabb.max.y):
                for u in range(triangle_aabb.min.x, triangle_aabb.max.x):
                    point = Vec2(int(u), int(v))
                    if (point.x < 0 or point.x >= GL.width) or (point.y < 0 or point.y >= GL.height) or not utils.is_point_inside_triangle(point, v0, v1, v2):
                        continue
                    
                    coefficients = utils.get_baricentric_coefficients(v0, v1, v2, point)
                    point_z = 1.0 / (coefficients.x / z0 + coefficients.y / z1 + coefficients.z / z2)

                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.depth_buffer)
                    if (point_z < gpu.GPU.read_pixel(point.to_list(), gpu.GPU.DEPTH_COMPONENT32F)):
                        gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.DEPTH_COMPONENT32F, [point_z])
                        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.draw_buffer)

                        point_color = Vec3()
                        if color_per_vertex:
                            c0 = Vec3()
                            c1 = Vec3()
                            c2 = Vec3()

                            c0.from_list(colors[(i // 3 * 9) + 0 : (i // 3 * 9) + 3])
                            c1.from_list(colors[(i // 3 * 9) + 3 : (i // 3 * 9) + 6])
                            c2.from_list(colors[(i // 3 * 9) + 6 : (i // 3 * 9) + 9])

                            c0 *= coefficients.x
                            c1 *= coefficients.y
                            c2 *= coefficients.z

                            point_color = (c0 + c1 + c2) * 255.0

                            gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.RGB8, np.clip(point_color.to_list(), 0, 255))
                        else:
                            transparency = colors["transparency"]
                            point_color.from_list(colors["emissiveColor"])

                            last_pixel_color = Vec3()
                            last_pixel_color.from_list(gpu.GPU.read_pixel(point.to_list(), gpu.GPU.RGB8))
                            last_pixel_color_transparent = last_pixel_color * transparency

                            point_color = point_color * (1.0 - transparency)
                            point_color += last_pixel_color_transparent
                            point_color *= 255.0

                            gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.RGB8, np.clip(point_color.to_list(), 0, 255))
                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.draw_buffer)

        # ps = []
        # zs = []
        # for i in range(0, len(pixelPoints), 3):
        #     zs.append(pixelPoints[i + 0][2])
        #     zs.append(pixelPoints[i + 1][2])
        #     zs.append(pixelPoints[i + 2][2])
        #     p0 = pixelPoints[i + 0][:2]
        #     p1 = pixelPoints[i + 1][:2]
        #     p2 = pixelPoints[i + 2][:2]
        #     ps = np.concatenate((ps, p0, p1, p2))
    
        # if type(colors) is dict:
        #     vertices = ps
        #     emissive_color = colors["emissive_color"]
        #     transparency = colors["transparency"]
        #     for i in range(0, len(vertices), 6):
        #         p0 = [vertices[i + 0], vertices[i + 1]]
        #         p1 = [vertices[i + 2], vertices[i + 3]]
        #         p2 = [vertices[i + 4], vertices[i + 5]]
        #         z0 = zs[i // 6 + 0]
        #         z1 = zs[i // 6 + 1]
        #         z2 = zs[i // 6 + 2]

        #         # Find the triangle AABB
        #         AABBMin = list(map(int, np.floor([min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1])])))
        #         AABBMax = list(map(int, np.ceil([max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1])])))
                
        #         # Fill the triangle
        #         for v in range(AABBMin[1], AABBMax[1]):
        #             for u in range(AABBMin[0], AABBMax[0]):
        #                 if (u < 0 or u >= GL.width) or (v < 0 or v >= GL.height):
        #                     continue

        #                 Q = [u, v]
        #                 P = [p1[0] - p0[0], p1[1] - p0[1]]
        #                 n = [P[1], -P[0]]
        #                 QLine = [Q[0] - p0[0], Q[1] - p0[1]]
        #                 dot = QLine[0] * n[0] + QLine[1] * n[1]
        #                 if dot < 0:
        #                     continue

        #                 P = [p2[0] - p1[0], p2[1] - p1[1]]
        #                 n = [P[1], -P[0]]
        #                 QLine = [Q[0] - p1[0], Q[1] - p1[1]]
        #                 dot = QLine[0] * n[0] + QLine[1] * n[1]
        #                 if dot < 0:
        #                     continue

        #                 P = [p0[0] - p2[0], p0[1] - p2[1]]
        #                 n = [P[1], -P[0]]
        #                 QLine = [Q[0] - p2[0], Q[1] - p2[1]]
        #                 dot = QLine[0] * n[0] + QLine[1] * n[1]
        #                 if dot < 0:
        #                     continue
                        
        #                 # Z interpolation
        #                 A = p0
        #                 B = p1
        #                 C = p2
        #                 p = [u, v]
        #                 alpha = (-(p[0] - B[0]) * (C[1] - B[1]) + (p[1] - B[1]) * (C[0] - B[0])) / (-(A[0] - B[0]) * (C[1] - B[1]) + (A[1] - B[1]) * (C[0] - B[0]))
        #                 beta = (-(p[0] - C[0]) * (A[1] - C[1]) + (p[1] - C[1]) * (A[0] - C[0])) / (-(B[0] - C[0]) * (A[1] - C[1]) + (B[1] - C[1]) * (A[0] - C[0]))
        #                 gamma = 1 - alpha - beta
                        
        #                 z = 1 / (alpha / z0 + beta / z1 + gamma / z2)

        #                 gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.depthBuffer)
        #                 if (z < gpu.GPU.read_pixel(p, gpu.GPU.DEPTH_COMPONENT32F)):
        #                     gpu.GPU.draw_pixel(p, gpu.GPU.DEPTH_COMPONENT32F, [z])
        #                     gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, [z * 255] * 3)
        #                     gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer)
                            
        #                     lastColor = gpu.GPU.read_pixel(p, gpu.GPU.RGB8)
        #                     lastColorTransparent = [c * transparency for c in lastColor]

        #                     newColor = [c * (1 - transparency) for c in emissive_color]
                            
        #                     bufferColor = [int(sum(c * 255)) for c in zip(newColor, lastColorTransparent)]
        #                     gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, np.clip(bufferColor, 0, 255))
        #                 gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer) 

        # else:
        #     vertices = ps
        #     for i in range(0, len(vertices), 6):
        #         p0 = [vertices[i + 0], vertices[i + 1]]
        #         p1 = [vertices[i + 2], vertices[i + 3]]
        #         p2 = [vertices[i + 4], vertices[i + 5]]
        #         z0 = zs[i // 6 + 0]
        #         z1 = zs[i // 6 + 1]
        #         z2 = zs[i // 6 + 2]
        #         c0 = colors[i + 0 + (3 * i // 6) : i + 3 + (3 * i // 6)]
        #         c1 = colors[i + 3  + (3 * i // 6): i + 6 + (3 * i // 6)]
        #         c2 = colors[i + 6  + (3 * i // 6): i + 9 + (3 * i // 6)]

        #         # Find the triangle AABB
        #         AABBMin = list(map(int, np.floor([min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1])])))
        #         AABBMax = list(map(int, np.ceil([max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1])])))
                
        #         # Fill the triangle
        #         for v in range(AABBMin[1], AABBMax[1]):
        #             for u in range(AABBMin[0], AABBMax[0]):
        #                 if (u < 0 or u >= GL.width) or (v < 0 or v >= GL.height):
        #                     continue

        #                 Q = [u, v]
        #                 P = [p1[0] - p0[0], p1[1] - p0[1]]
        #                 n = [P[1], -P[0]]
        #                 QLine = [Q[0] - p0[0], Q[1] - p0[1]]
        #                 dot = QLine[0] * n[0] + QLine[1] * n[1]
        #                 if dot < 0:
        #                     continue

        #                 P = [p2[0] - p1[0], p2[1] - p1[1]]
        #                 n = [P[1], -P[0]]
        #                 QLine = [Q[0] - p1[0], Q[1] - p1[1]]
        #                 dot = QLine[0] * n[0] + QLine[1] * n[1]
        #                 if dot < 0:
        #                     continue

        #                 P = [p0[0] - p2[0], p0[1] - p2[1]]
        #                 n = [P[1], -P[0]]
        #                 QLine = [Q[0] - p2[0], Q[1] - p2[1]]
        #                 dot = QLine[0] * n[0] + QLine[1] * n[1]
        #                 if dot < 0:
        #                     continue
                        
        #                 # Color interpolation
        #                 A = p0
        #                 B = p1
        #                 C = p2
        #                 p = [u, v]
        #                 alpha = (-(p[0] - B[0]) * (C[1] - B[1]) + (p[1] - B[1]) * (C[0] - B[0])) / (-(A[0] - B[0]) * (C[1] - B[1]) + (A[1] - B[1]) * (C[0] - B[0]))
        #                 beta = (-(p[0] - C[0]) * (A[1] - C[1]) + (p[1] - C[1]) * (A[0] - C[0])) / (-(B[0] - C[0]) * (A[1] - C[1]) + (B[1] - C[1]) * (A[0] - C[0]))
        #                 gamma = 1 - alpha - beta
                        
        #                 z = 1 / (alpha / z0 + beta / z1 + gamma / z2)

        #                 gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.depthBuffer) 
        #                 if (z < gpu.GPU.read_pixel(p, gpu.GPU.DEPTH_COMPONENT32F)):
        #                     gpu.GPU.draw_pixel(p, gpu.GPU.DEPTH_COMPONENT32F, [z])
        #                     gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, [z * 255] * 3)

        #                     _c0 = [c * alpha for c in c0]
        #                     _c1 = [c * beta for c in c1]
        #                     _c2 = [c * gamma for c in c2]
        #                     Cs = [_c0, _c1, _c2]
        #                     _color = [sum(c * 255) for c in zip(*Cs)]
                            
        #                     gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer) 
        #                     gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, _color)
        #                 gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer) 

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""

        eye = Vec3(position[0], position[1], position[2])
        _orientation = Vec4(orientation[0], orientation[1], orientation[2], orientation[3])

        aspect_ratio = GL.width / GL.height
        fovy = 2 * np.arctan(np.tan(fieldOfView / 2.0) * GL.height / np.sqrt(GL.height**2 + GL.width**2))

        screen_matrix = Mat4(identity = True)
        screen_matrix[0] = Vec4(GL.width / 2, 0, 0, GL.width / 2)
        screen_matrix[1] = Vec4(0, -GL.height / 2, 0, GL.height / 2)
        screen_matrix[2] = Vec4(0, 0, 1, 0)
        screen_matrix[3] = Vec4(0, 0, 0, 1)
        print(screen_matrix, "\n")

        camera_matrix = Mat4(identity = True)
        camera_matrix = camera_matrix.look_at(eye, _orientation)
        camera_matrix = camera_matrix.perspective(fovy, aspect_ratio, GL.near, GL.far)
        camera_matrix = screen_matrix * camera_matrix

        GL.viewpoint_matrix = camera_matrix

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""

        _translation = Vec3(translation[0], translation[1], translation[2])
        _scale = Vec3(scale[0], scale[1], scale[2])
        rotation_angle = rotation[3]
        rotation_axis = Vec3(rotation[0], rotation[1], rotation[2])
        rotation_axis.normalize()
        
        model_matrix = Mat4(identity = True)
        model_matrix = model_matrix.translate(_translation)
        model_matrix = model_matrix.rotate(rotation_angle, rotation_axis)
        model_matrix = model_matrix.scale(_scale)

        parent_model_matrix = GL.transform_stack.peek()
        relative_model_matrix = parent_model_matrix * model_matrix

        GL.transform_stack.push(relative_model_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""

        GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        ps = []
        for i in range(stripCount[0] - 2):
            p0 = i + 0
            p1 = i + 1
            p2 = i + 2
            triangle_points = [
                point[p0 * 3 : p0 * 3 + 3],
                point[p1 * 3 : p1 * 3 + 3],
                point[p2 * 3 : p2 * 3 + 3],
            ]

            if (i + 1) % 2 == 0:
                ps.extend([*triangle_points[1], *triangle_points[0], *triangle_points[2]])
            else:
                ps.extend([*triangle_points[0], *triangle_points[1], *triangle_points[2]])

        GL.triangleSet(ps, colors)

        ps = []
        for i in range(stripCount[0] - 6):
            p0 = point[i + 0 : i + 3]
            p1 = point[i + 3 : i + 6]
            p2 = point[i + 6 : i + 9]
            if (i + 1) % 2 == 0:
                ps.extend([*p1, *p0, *p2])
            else:
                ps.extend([*p0, *p1, *p2])
        GL.triangleSet(ps, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        ps = []
        for i in range(len(index) - 3):
            p0 = index[i + 0]
            p1 = index[i + 1]
            p2 = index[i + 2]
            triangle_points = [
                point[p0 * 3 : p0 * 3 + 3],
                point[p1 * 3 : p1 * 3 + 3],
                point[p2 * 3 : p2 * 3 + 3],
            ]

            if (i + 1) % 2 == 0:
                ps.extend([*triangle_points[1], *triangle_points[0], *triangle_points[2]])
            else:
                ps.extend([*triangle_points[0], *triangle_points[1], *triangle_points[2]])

        GL.triangleSet(ps, colors)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens. 

        ps = []
        for i in range(0, len(coordIndex) - 3, 4):
            p0 = coordIndex[i + 0]
            p1 = coordIndex[i + 1]
            p2 = coordIndex[i + 2]
            triangle_points = [
                coord[p0 * 3 : p0 * 3 + 3],
                coord[p1 * 3 : p1 * 3 + 3],
                coord[p2 * 3 : p2 * 3 + 3]
            ]
            if (i + 1) % 2 == 0:
                ps.extend([*triangle_points[1], *triangle_points[0], *triangle_points[2]])
            else:
                ps.extend([*triangle_points[0], *triangle_points[1], *triangle_points[2]])

        if len(colorIndex) == 0:
            GL.triangleSet(ps, colors)
        else:
            cs = []
            for i in range (0, len(colorIndex) - 3, 4):
                c0 = colorIndex[i + 0]
                c1 = colorIndex[i + 1]
                c2 = colorIndex[i + 2]
                triangle_colors = [ 
                    color[c0 * 3 : c0 * 3 + 3],
                    color[c1 * 3 : c1 * 3 + 3],
                    color[c2 * 3 : c2 * 3 + 3],
                ]
                if (i + 1) % 2 == 0:
                    cs.extend([*triangle_colors[1], *triangle_colors[0], *triangle_colors[2]])
                else:
                    cs.extend([*triangle_colors[0], *triangle_colors[1], *triangle_colors[2]])
            GL.triangleSet(ps, cs)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
