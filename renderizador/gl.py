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
import random

import utils

from stack import Stack
from texture import Texture
from light import Light

from Math.vec2 import *
from Math.vec3 import *
from Math.mat4 import *
from Math.aabb import *
from Math.vec_math import *

import Math.lin_alg as lin_alg

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    
    transform_stack = Stack()
    transform_stack.push(Mat4(identity = True))

    viewpoint_matrix = Mat4(identity = True)

    camera_position = Vec3()

    draw_buffer = None
    depth_buffer = None

    current_x3d_path = None

    target_texture = None

    lights = list()

    sphere_refinement_level = 2

    debug_colors = False

    @staticmethod
    def setup(width, height, draw_buffer, depth_buffer, current_x3d_path, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.draw_buffer = draw_buffer
        GL.depth_buffer = depth_buffer
        GL.current_x3d_path = current_x3d_path

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

        emissive_color = [int(255 * c) for c in colors["emissiveColor"]]
        for i in range(0, len(vertices), 6):
            v0 = Vec2(vertices[i + 0], vertices[i + 1])
            v1 = Vec2(vertices[i + 2], vertices[i + 3])
            v2 = Vec2(vertices[i + 4], vertices[i + 5])

            # Find the triangle AABB
            triangle_aabb = AABB()
            triangle_aabb.create_from_triangle(v0, v1, v2)
            
            # Fill the triangle
            for v in range(int(triangle_aabb.min.y), int(triangle_aabb.max.y)):
                for u in range(int(triangle_aabb.min.x), int(triangle_aabb.max.x)):
                    point = Vec2(u, v)
                    if (point.x < 0 or point.x >= GL.width) or (point.y < 0 or point.y >= GL.height):
                        continue

                    if not utils.is_point_inside_triangle(point, v0, v1, v2):
                        continue

                    gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.RGB8, emissive_color)

    @staticmethod
    def triangleSet(point, colors, texCoord = None):
        """Função usada para renderizar TriangleSet."""
        
        # Tive que usar Numpy por que não tive paciência de fazer uma classe de matriz genérica. Talvez eu faça depois...
        points_matrix = list()
        for i in range(0, len(point), 3):
            points_matrix.append([point[i + 0], point[i + 1], point[i + 2], 1])
        points_matrix = np.transpose(np.array(points_matrix))

        model_matrix = GL.transform_stack.peek()
        transformed_points = np.matmul(model_matrix.to_list(), points_matrix)
        
        transformed_points = np.transpose(transformed_points)
        triangles_normals = utils.get_triangles_normals(transformed_points)
        transformed_points = np.transpose(transformed_points)

        viewpoint_matrix = GL.viewpoint_matrix
        projected_points = np.matmul(viewpoint_matrix.to_list(), transformed_points)

        projected_points = utils.homogeneous_divide(projected_points)
        projected_points = np.transpose(projected_points)

        # Rasterize the triangles
        point_list = list()
        z_list = list()

        for point in projected_points:
            point_list.append(Vec2(point[0], point[1]))
            z_list.append(point[2])

        color_per_vertex = True if isinstance(colors, list) else False
        texture_mapping = True if texCoord else False

        screen_aabb = AABB(Vec2(), Vec2(GL.width, GL.height))

        # Material properties
        emissive_color = Vec3()
        diffuse_color = Vec3()
        specular_color = Vec3()
        ambient_factor = 0.0
        specular_factor = 0.0
        transparency = 0.0

        if not color_per_vertex and not texture_mapping:
            emissive_color.from_list(colors["emissiveColor"])
            diffuse_color.from_list(colors["diffuseColor"])
            specular_color.from_list(colors["specularColor"])
            ambient_factor = colors["ambientIntensity"] if "ambientIntensity" in colors else 0.2
            specular_factor = colors["shininess"] if "shininess" in colors else 0.2
            transparency = colors["transparency"] if "transparency" in colors else 0.0

        material = {
            "emissive_color": emissive_color,
            "diffuse_color": diffuse_color,
            "specular_color": specular_color,
            "ambient_factor": ambient_factor,
            "specular_factor": specular_factor,
            "transparency": transparency
        }

        # Triangles rasterization
        for i in range(0, len(point_list), 3):
            v0 = point_list[i + 0]
            v1 = point_list[i + 1]
            v2 = point_list[i + 2]

            z0 = z_list[i + 0]
            z1 = z_list[i + 1]
            z2 = z_list[i + 2]

            vn = triangles_normals[i // 3]
            
            if GL.debug_colors and not color_per_vertex and not texCoord:
                material["emissive_color"] = Vec3(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0))
                material["diffuse_color"] = Vec3(random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0))

            triangle_aabb = AABB()
            triangle_aabb.create_from_triangle(v0, v1, v2)
            
            if not triangle_aabb.intersects(screen_aabb):
                continue

            intersection_aabb = triangle_aabb.intersection_square(screen_aabb)
            
            for v in range(int(intersection_aabb.min.y), int(intersection_aabb.max.y)):
                for u in range(int(intersection_aabb.min.x), int(intersection_aabb.max.x)):
                    point = Vec2(u, v)
                    if (point.x < 0 or point.x >= GL.width) or (point.y < 0 or point.y >= GL.height):
                        continue

                    if not utils.is_point_inside_triangle(point, v0, v1, v2):
                        continue
                    
                    coefficients = utils.get_baricentric_coefficients(v0, v1, v2, point)
                    point_x = 1.0 / (coefficients.x / v0.x + coefficients.y / v1.x + coefficients.z / v2.x)
                    point_y = 1.0 / (coefficients.x / v0.y + coefficients.y / v1.y + coefficients.z / v2.y)
                    point_z = 1.0 / (coefficients.x / z0 + coefficients.y / z1 + coefficients.z / z2)

                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.depth_buffer)
                    if (point_z < gpu.GPU.read_pixel(point.to_list(), gpu.GPU.DEPTH_COMPONENT32F)):
                        gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.DEPTH_COMPONENT32F, [point_z])
                        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.draw_buffer)

                        pixel_color = Vec3()
                        if color_per_vertex and not texture_mapping:
                            c0 = Vec3()
                            c1 = Vec3()
                            c2 = Vec3()

                            c0.from_list(colors[(i // 3 * 9) + 0 : (i // 3 * 9) + 3])
                            c1.from_list(colors[(i // 3 * 9) + 3 : (i // 3 * 9) + 6])
                            c2.from_list(colors[(i // 3 * 9) + 6 : (i // 3 * 9) + 9])

                            c0 *= coefficients.x
                            c1 *= coefficients.y
                            c2 *= coefficients.z

                            pixel_color = (c0 + c1 + c2)
                        elif texture_mapping:
                            uv0 = texCoord[i + 0] 
                            uv1 = texCoord[i + 1]
                            uv2 = texCoord[i + 2]

                            uv0 *= coefficients.x
                            uv1 *= coefficients.y
                            uv2 *= coefficients.z

                            pixel_color = GL.target_texture.read_pixel(uv0 + uv1 + uv2)
                            pixel_color /= 255.0
                        elif len(GL.lights) == 0:
                            pixel_color = material["emissive_color"]
                            transparency = material["transparency"]
                            if (transparency > 0.0):
                                last_pixel_color = Vec3()
                                last_pixel_color.from_list(gpu.GPU.read_pixel(point.to_list(), gpu.GPU.RGB8))
                                last_pixel_color_transparent = last_pixel_color * transparency

                                pixel_color *= 1.0 - transparency
                                pixel_color += last_pixel_color_transparent
                        else:
                            material_ambient_factor = material["ambient_factor"]
                            material_specular_factor = material["specular_factor"]
                            material_emissive_color = material["emissive_color"]
                            material_diffuse_color = material["diffuse_color"]
                            material_specular_color = material["specular_color"]
                            
                            eye = GL.camera_position - Vec3(point_x, point_y, point_z)
                            eye.normalize()
                            
                            p = material_specular_factor * 128
                            for light in GL.lights:
                                if light.is_directional:
                                    H = (light.direction * -1.0) - vn * 2.0 * dot(vn, (light.direction * -1.0))

                                    pixel_ambient_color = material_diffuse_color * light.ambient_intensity * material_ambient_factor
                                    pixel_diffuse_color = material_diffuse_color * light.intensity * max(0.0, dot(vn, light.direction)) 
                                    pixel_specular_color = material_specular_color * light.intensity * max(0.0, dot(eye, H))**p
                                    
                                    pixel_color += material_emissive_color + (light.color * (pixel_ambient_color + pixel_diffuse_color + pixel_specular_color)) 
                        pixel_color *= 255.0
                        gpu.GPU.draw_pixel(point.to_list(), gpu.GPU.RGB8, np.clip(pixel_color.to_list(), 0.0, 255.0))
                    gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.draw_buffer) 

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""

        eye = Vec3(position[0], position[1], position[2])
        _orientation = Vec4(orientation[0], orientation[1], orientation[2], orientation[3])
        look_at_matrix = lin_alg.look_at(eye, _orientation)

        aspect_ratio = GL.width / GL.height
        fovy = 2 * np.arctan(np.tan(fieldOfView / 2.0) * GL.height / np.sqrt(GL.height**2 + GL.width**2))
        perspective_matrix = lin_alg.perspective(fovy, aspect_ratio, GL.near, GL.far)

        screen_matrix = Mat4(identity = True)
        screen_matrix[0] = Vec4(GL.width / 2, 0, 0, GL.width / 2)
        screen_matrix[1] = Vec4(0, -GL.height / 2, 0, GL.height / 2)
        screen_matrix[2] = Vec4(0, 0, 1, 0)
        screen_matrix[3] = Vec4(0, 0, 0, 1)

        camera_matrix = screen_matrix * perspective_matrix * look_at_matrix
        GL.camera_position = eye
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
        model_matrix = lin_alg.scale(model_matrix, _scale)
        model_matrix = lin_alg.rotate(model_matrix, rotation_angle, rotation_axis)
        model_matrix = lin_alg.translate(model_matrix, _translation)

        GL.transform_stack.push(GL.transform_stack.peek() * model_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""

        GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""

        vertices = list()
        for i in range(0, len(point), 3):
            vertices.append(Vec3(point[i + 0], point[i + 1], point[i + 2]))
        
        offset = 0
        draw_vertices = list()
        for strip in stripCount:
            for i in range(strip - 2):
                if (i + 1) % 2 == 0:
                    draw_vertices.extend(vertices[i + offset + 1].to_list())
                    draw_vertices.extend(vertices[i + offset + 0].to_list())
                    draw_vertices.extend(vertices[i + offset + 2].to_list())
                else:
                    draw_vertices.extend(vertices[i + offset + 0].to_list())
                    draw_vertices.extend(vertices[i + offset + 1].to_list())
                    draw_vertices.extend(vertices[i + offset + 2].to_list())                 
            offset += strip
        GL.triangleSet(draw_vertices, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""

        vertices = list()
        for i in range(0, len(point), 3):
            vertices.append(Vec3(point[i + 0], point[i + 1], point[i + 2]))

        draw_vertices = list()
        for i in range(len(index) - 2):
            if (i + 1) % 2 == 0:
                draw_vertices.extend(vertices[index[i + 1]].to_list())
                draw_vertices.extend(vertices[index[i + 0]].to_list())
                draw_vertices.extend(vertices[index[i + 2]].to_list())
            else:
                draw_vertices.extend(vertices[index[i + 0]].to_list())
                draw_vertices.extend(vertices[index[i + 1]].to_list())
                draw_vertices.extend(vertices[index[i + 2]].to_list())

        GL.triangleSet(draw_vertices, colors)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        default_box_vertices = [
            Vec3(-0.5, -0.5, -0.5),
            Vec3( 0.5, -0.5, -0.5),
            Vec3( 0.5,  0.5, -0.5),
            Vec3(-0.5,  0.5, -0.5),
            Vec3(-0.5, -0.5,  0.5),
            Vec3( 0.5, -0.5,  0.5),
            Vec3( 0.5,  0.5,  0.5),
            Vec3(-0.5,  0.5,  0.5)
        ]

        default_box_indices = [
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
            3, 2, 1, 3, 1, 0,
            4, 5, 6, 4, 6, 7
        ]

        for vertex in default_box_vertices:
            vertex.x *= size[0]
            vertex.y *= size[1]
            vertex.z *= size[2]

        draw_vertices = list()
        for index in default_box_indices:
            draw_vertices.extend(default_box_vertices[index].to_list())

        GL.triangleSet(draw_vertices, colors)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet.""" 

        vertices = list()
        for i in range(0, len(coord), 3):
            vertices.append(Vec3(coord[i + 0], coord[i + 1], coord[i + 2]))

        offset = 0
        draw_vertices = list()
        for i in range(len(coordIndex) - 2):
            if coordIndex[i + offset + 2] == -1:
                offset += 3

            if i + offset >= len(coordIndex):
                break

            draw_vertices.extend(vertices[coordIndex[i + offset + 0]].to_list())
            draw_vertices.extend(vertices[coordIndex[i + offset + 1]].to_list())
            draw_vertices.extend(vertices[coordIndex[i + offset + 2]].to_list()) 

        # No color per vertex and no texture
        if len(colorIndex) == 0 and len(texCoordIndex) == 0:
            GL.triangleSet(draw_vertices, colors)

        # Color per vertex
        elif len(colorIndex) > 0:
            _colors = list()
            for i in range(0, len(color), 3):
                _colors.append(Vec3(color[i + 0], color[i + 1], color[i + 2]))
            
            i = 0
            vertex_colors = list()
            for _ in range(len(colorIndex) - 2):
                if coordIndex[i + 2] == -1:
                    i += 3
            
                if i > len(coordIndex) - 2:
                    break
                
                if (i + 1) % 2 == 0:
                    vertex_colors.extend(_colors[colorIndex[i + 1]].to_list())
                    vertex_colors.extend(_colors[colorIndex[i + 0]].to_list())
                    vertex_colors.extend(_colors[colorIndex[i + 2]].to_list())
                else:
                    vertex_colors.extend(_colors[colorIndex[i + 0]].to_list())
                    vertex_colors.extend(_colors[colorIndex[i + 1]].to_list())
                    vertex_colors.extend(_colors[colorIndex[i + 2]].to_list())
                
                i += 1

            GL.triangleSet(draw_vertices, vertex_colors)
            
        # Texture
        elif len(texCoordIndex) > 0:
            _texCoord = list()
            for i in range(0, len(texCoord), 2):
                _texCoord.append(Vec2(texCoord[i + 0], texCoord[i + 1]))

            offset = 0
            vertex_uvs = list()
            for i in range(len(texCoordIndex) - 2):
                if texCoordIndex[i + offset + 2] == -1:
                    offset += 3
            
                if i + offset >= len(texCoordIndex):
                    break
                
                vertex_uvs.append(_texCoord[texCoordIndex[i + offset + 0]])
                vertex_uvs.append(_texCoord[texCoordIndex[i + offset + 1]])
                vertex_uvs.append(_texCoord[texCoordIndex[i + offset + 2]])

            GL.target_texture = Texture(GL.current_x3d_path + "/" + current_texture[0], True)
            GL.triangleSet(draw_vertices, [], vertex_uvs)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""

        ico_radius = (1.0 + np.sqrt(5.0)) / 2.0
        ico_vertices = [
            Vec3(-1.0,  ico_radius, 0.0),
            Vec3( 1.0,  ico_radius, 0.0),
            Vec3(-1.0, -ico_radius, 0.0),
            Vec3( 1.0, -ico_radius, 0.0),
            Vec3( 0.0, -1.0,   ico_radius),
            Vec3( 0.0,  1.0,   ico_radius),
            Vec3( 0.0, -1.0,  -ico_radius),
            Vec3( 0.0,  1.0,  -ico_radius),
            Vec3( ico_radius, 0.0, -1.0),
            Vec3( ico_radius, 0.0,  1.0),
            Vec3(-ico_radius, 0.0, -1.0),
            Vec3(-ico_radius, 0.0,  1.0),
        ]

        for i in range(len(ico_vertices)):
            ico_vertices[i].normalize()
            ico_vertices[i] *= radius

        ico_indices = [
             0, 11,  5, 0,  5,  1,
             0,  1,  7, 0,  7, 10,
             0, 10, 11, 1,  5,  9,
             5, 11,  4, 11, 10,  2,
            10,  7,  6, 7,  1,  8,
             3,  9,  4, 3,  4,  2,
             3,  2,  6, 3,  6,  8,
             3,  8,  9, 5,  4,  9,
             2,  4, 11, 6,  2, 10,
             8,  6,  7, 9,  8,  1,
        ]

        for i in range(GL.sphere_refinement_level):
            for i in range(0, len(ico_indices), 3):
                v0 = ico_vertices[ico_indices[i + 0]]
                v1 = ico_vertices[ico_indices[i + 1]]
                v2 = ico_vertices[ico_indices[i + 2]]

                v0_v1_middle = v0 + v1
                scale = radius / v0_v1_middle.length()
                v0_v1_middle *= scale
                ico_vertices.append(v0_v1_middle)

                v0_v1_index = len(ico_vertices) - 1

                v1_v2_middle = v1 + v2
                scale = radius / v1_v2_middle.length()
                v1_v2_middle *= scale
                ico_vertices.append(v1_v2_middle)

                v1_v2_index = len(ico_vertices) - 1

                v2_v0_middle = v2 + v0
                scale = radius / v2_v0_middle.length()
                v2_v0_middle *= scale
                ico_vertices.append(v2_v0_middle)

                v2_v0_index = len(ico_vertices) - 1

                ico_indices.extend([ico_indices[i + 0], v0_v1_index, v2_v0_index])
                ico_indices.extend([ico_indices[i + 1], v1_v2_index, v0_v1_index])
                ico_indices.extend([ico_indices[i + 2], v2_v0_index, v1_v2_index])
                ico_indices.extend([v0_v1_index, v1_v2_index, v2_v0_index])

        draw_vertices = list()
        for index in ico_indices:
            draw_vertices.extend(ico_vertices[index].to_list())

        GL.triangleSet(draw_vertices, colors)

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        
        if headlight:
            light_direction = Vec3(0, 0, -1)
            light_color = Vec3(1, 1, 1)
            head_light = Light(is_directional = True, direction = light_direction, color = light_color)

            GL.lights.append(head_light)

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""

        light_direction = Vec3()
        light_color = Vec3()
        
        light_direction.from_list(direction)
        light_direction.normalize()

        light_color.from_list(color)

        directional_light = Light(is_directional = True, direction = light_direction, color = light_color, intensity = intensity, ambient_intensity = ambientIntensity)
        GL.lights.append(directional_light)

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

        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""

        first_index     = np.searchsorted(key, set_fraction)
        second_index    = (first_index + 1) % len(key)
        third_index     = (first_index + -1) % len(key)
        fourth_index    = (first_index + 2) % len(key)
        t = set_fraction / (key[1] - key[0]) % 1

        t_values            = np.array([t**3, t**2, t, 1.0])
        curve_space_matrix  = np.array([[-0.5,  1.5, -1.5,  0.5],
                                        [ 1. , -2.5,  2. , -0.5],
                                        [-0.5,  0. ,  0.5,  0. ],
                                        [ 0. ,  1. ,  0. ,  0. ]])
        
        p0 = keyValue[third_index   * 3 : third_index   * 3 + 3]
        p1 = keyValue[first_index   * 3 : first_index   * 3 + 3]
        p2 = keyValue[second_index  * 3 : second_index  * 3 + 3]
        p3 = keyValue[fourth_index  * 3 : fourth_index  * 3 + 3]
        x_values = np.transpose(np.array([p0[0], p1[0], p2[0], p3[0]]))
        y_values = np.transpose(np.array([p0[1], p1[1], p2[1], p3[1]]))
        z_values = np.transpose(np.array([p0[2], p1[2], p2[2], p3[2]]))

        x = np.matmul(t_values, np.matmul(curve_space_matrix, x_values))
        y = np.matmul(t_values, np.matmul(curve_space_matrix, y_values))
        z = np.matmul(t_values, np.matmul(curve_space_matrix, z_values))

        return [x, y, z]

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""

        first_index     = int(set_fraction / (key[1] - key[0]))
        second_index    = (first_index + 1) % len(key)
        third_index     = (first_index + 2) % len(key)
        fourth_index    = (first_index + 3) % len(key)
        t = set_fraction / (key[1] - key[0]) % 1

        t_values            = np.array([t**3, t**2, t, 1.0])
        curve_space_matrix  = np.array([[ 0.5,  1.5, -1.5,  0.5],
                                        [ 0. , -2.5,  2. , -0.5],
                                        [-0.5,  0. ,  0.5,  0. ],
                                        [ 0. ,  1. ,  0. ,  0. ]])
        
        p0 = keyValue[first_index * 4   : first_index * 4 + 4]
        p1 = keyValue[second_index * 4  : second_index * 4 + 4]
        p2 = keyValue[third_index * 4   : third_index * 4 + 4]
        p3 = keyValue[fourth_index * 4  : fourth_index * 4 + 4]
        x_values = np.transpose(np.array([p0[0], p1[0], p2[0], p3[0]]))
        y_values = np.transpose(np.array([p0[1], p1[1], p2[1], p3[1]]))
        z_values = np.transpose(np.array([p0[2], p1[2], p2[2], p3[2]]))
        w_values = np.transpose(np.array([p0[3], p1[3], p2[3], p3[3]]))

        x = np.matmul(t_values, np.matmul(curve_space_matrix, x_values))
        y = np.matmul(t_values, np.matmul(curve_space_matrix, y_values))
        z = np.matmul(t_values, np.matmul(curve_space_matrix, z_values))
        w = np.matmul(t_values, np.matmul(curve_space_matrix, w_values))

        return [x, y, z, w]

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
