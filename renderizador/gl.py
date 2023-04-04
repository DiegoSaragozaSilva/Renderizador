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

from stack import Stack

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    
    transformStack = Stack()
    transformStack.push(np.identity(4))

    projectionBuffer = []

    drawBuffer = None
    depthBuffer = None

    @staticmethod
    def setup(width, height, drawBuffer, depthBuffer, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.drawBuffer = drawBuffer
        GL.depthBuffer = depthBuffer

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        emissiveColor = [int(255 * c) for c in colors["emissiveColor"]]
        for i in range(0, len(point), 2):
            u = int(point[i + 0])
            v = int(point[i + 1])
            gpu.GPU.draw_pixel([u, v], gpu.GPU.RGB8, emissiveColor)

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        emissiveColor = [int(255 * c) for c in colors["emissiveColor"]]

        # Bresenham's algorithm with error correction
        for i in range(0, len(lineSegments) - 2, 2):
            uStart = int(lineSegments[i + 0])
            vStart = int(lineSegments[i + 1])
            uEnd   = int(lineSegments[i + 2])
            vEnd   = int(lineSegments[i + 3])

            u = uStart
            v = vStart
            du = abs(uEnd - uStart)
            dv = abs(vEnd - vStart)
            su = 1 if uStart < uEnd else -1
            sv = 1 if vStart < vEnd else -1
            error = du if du > dv else -dv
            error /= 2
            lastError = 0
            
            while True:
                if (u >= 0 and v >= 0 and u < GL.width and v < GL.height):
                    gpu.GPU.draw_pixel([u, v], gpu.GPU.RGB8, emissiveColor)
                
                if (u == uEnd and v == vEnd):
                    break
                lastError = error
                if (lastError > -du):
                    error -= dv
                    u += su
                if (lastError < dv):
                    error += du
                    v += sv

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D

        emissiveColor = [int(255 * c) for c in colors["emissiveColor"]]
        for i in range(0, len(vertices), 6):
            p0 = [vertices[i + 0], vertices[i + 1]]
            p1 = [vertices[i + 2], vertices[i + 3]]
            p2 = [vertices[i + 4], vertices[i + 5]]

            # Find the triangle AABB
            AABBMin = list(map(int, np.floor([min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1])])))
            AABBMax = list(map(int, np.ceil([max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1])])))
            
            # Fill the triangle
            for v in range(AABBMin[1], AABBMax[1]):
                for u in range(AABBMin[0], AABBMax[0]):
                    if (u < 0 or u >= GL.width) or (v < 0 or v >= GL.height):
                        continue

                    Q = [u, v]
                    P = [p1[0] - p0[0], p1[1] - p0[1]]
                    n = [P[1], -P[0]]
                    QLine = [Q[0] - p0[0], Q[1] - p0[1]]
                    dot = QLine[0] * n[0] + QLine[1] * n[1]
                    if dot < 0:
                        continue

                    P = [p2[0] - p1[0], p2[1] - p1[1]]
                    n = [P[1], -P[0]]
                    QLine = [Q[0] - p1[0], Q[1] - p1[1]]
                    dot = QLine[0] * n[0] + QLine[1] * n[1]
                    if dot < 0:
                        continue

                    P = [p0[0] - p2[0], p0[1] - p2[1]]
                    n = [P[1], -P[0]]
                    QLine = [Q[0] - p2[0], Q[1] - p2[1]]
                    dot = QLine[0] * n[0] + QLine[1] * n[1]
                    if dot < 0:
                        continue

                    gpu.GPU.draw_pixel([u, v], gpu.GPU.RGB8, emissiveColor)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        
        # Transformations 
        pointsMatrix = []
        for i in range(0, len(point), 3):
            pointsMatrix.append([point[i + 0], point[i + 1], point[i + 2], 1])
        pointsMatrix = np.transpose(np.array(pointsMatrix))
        
        # Transform
        modelMatrix = GL.transformStack.peek()
        transformedPoints = np.matmul(modelMatrix, pointsMatrix)

        # Project
        projectionMatrix = GL.projectionBuffer[len(GL.projectionBuffer) - 1]
        projectedPoints = np.transpose(np.matmul(projectionMatrix, transformedPoints)) 

        # Divide by w
        for i in range(len(projectedPoints)):
            projectedPoint = projectedPoints[i] 
            projectedPoint[0] /= projectedPoint[3]
            projectedPoint[1] /= projectedPoint[3]
            projectedPoint[2] /= projectedPoint[3]
            projectedPoint[3] = 1
            projectedPoints[i] = projectedPoint
        projectedPoints = np.transpose(projectedPoints)

        # Camera normalized space to pixel space
        screenMatrix = np.array([[GL.width / 2, 0, 0, GL.width / 2],
                                 [0, -GL.height / 2, 0, GL.height / 2],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        pixelPoints = np.transpose(np.matmul(screenMatrix, projectedPoints))

        # Rasterize points
        ps = []
        zs = []
        for i in range(0, len(pixelPoints), 3):
            zs.append(pixelPoints[i + 0][2])
            zs.append(pixelPoints[i + 1][2])
            zs.append(pixelPoints[i + 2][2])
            p0 = pixelPoints[i + 0][:2]
            p1 = pixelPoints[i + 1][:2]
            p2 = pixelPoints[i + 2][:2]
            ps = np.concatenate((ps, p0, p1, p2))
    
        if type(colors) is dict:
            vertices = ps
            emissiveColor = colors["emissiveColor"]
            transparency = colors["transparency"]
            for i in range(0, len(vertices), 6):
                p0 = [vertices[i + 0], vertices[i + 1]]
                p1 = [vertices[i + 2], vertices[i + 3]]
                p2 = [vertices[i + 4], vertices[i + 5]]
                z0 = zs[i // 6 + 0]
                z1 = zs[i // 6 + 1]
                z2 = zs[i // 6 + 2]

                # Find the triangle AABB
                AABBMin = list(map(int, np.floor([min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1])])))
                AABBMax = list(map(int, np.ceil([max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1])])))
                
                # Fill the triangle
                for v in range(AABBMin[1], AABBMax[1]):
                    for u in range(AABBMin[0], AABBMax[0]):
                        if (u < 0 or u >= GL.width) or (v < 0 or v >= GL.height):
                            continue

                        Q = [u, v]
                        P = [p1[0] - p0[0], p1[1] - p0[1]]
                        n = [P[1], -P[0]]
                        QLine = [Q[0] - p0[0], Q[1] - p0[1]]
                        dot = QLine[0] * n[0] + QLine[1] * n[1]
                        if dot < 0:
                            continue

                        P = [p2[0] - p1[0], p2[1] - p1[1]]
                        n = [P[1], -P[0]]
                        QLine = [Q[0] - p1[0], Q[1] - p1[1]]
                        dot = QLine[0] * n[0] + QLine[1] * n[1]
                        if dot < 0:
                            continue

                        P = [p0[0] - p2[0], p0[1] - p2[1]]
                        n = [P[1], -P[0]]
                        QLine = [Q[0] - p2[0], Q[1] - p2[1]]
                        dot = QLine[0] * n[0] + QLine[1] * n[1]
                        if dot < 0:
                            continue
                        
                        # Z interpolation
                        A = p0
                        B = p1
                        C = p2
                        p = [u, v]
                        alpha = (-(p[0] - B[0]) * (C[1] - B[1]) + (p[1] - B[1]) * (C[0] - B[0])) / (-(A[0] - B[0]) * (C[1] - B[1]) + (A[1] - B[1]) * (C[0] - B[0]))
                        beta = (-(p[0] - C[0]) * (A[1] - C[1]) + (p[1] - C[1]) * (A[0] - C[0])) / (-(B[0] - C[0]) * (A[1] - C[1]) + (B[1] - C[1]) * (A[0] - C[0]))
                        gamma = 1 - alpha - beta
                        
                        z = 1 / (alpha / z0 + beta / z1 + gamma / z2)

                        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.depthBuffer)
                        if (z < gpu.GPU.read_pixel(p, gpu.GPU.DEPTH_COMPONENT32F)):
                            gpu.GPU.draw_pixel(p, gpu.GPU.DEPTH_COMPONENT32F, [z])
                            gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, [z * 255] * 3)
                            gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer)
                            
                            lastColor = gpu.GPU.read_pixel(p, gpu.GPU.RGB8)
                            lastColorTransparent = [c * transparency for c in lastColor]

                            newColor = [c * (1 - transparency) for c in emissiveColor]
                            
                            bufferColor = [int(sum(c * 255)) for c in zip(newColor, lastColorTransparent)]
                            gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, np.clip(bufferColor, 0, 255))
                        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer) 

        else:
            vertices = ps
            for i in range(0, len(vertices), 6):
                p0 = [vertices[i + 0], vertices[i + 1]]
                p1 = [vertices[i + 2], vertices[i + 3]]
                p2 = [vertices[i + 4], vertices[i + 5]]
                z0 = zs[i // 6 + 0]
                z1 = zs[i // 6 + 1]
                z2 = zs[i // 6 + 2]
                c0 = colors[i + 0 + (3 * i // 6) : i + 3 + (3 * i // 6)]
                c1 = colors[i + 3  + (3 * i // 6): i + 6 + (3 * i // 6)]
                c2 = colors[i + 6  + (3 * i // 6): i + 9 + (3 * i // 6)]

                # Find the triangle AABB
                AABBMin = list(map(int, np.floor([min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1])])))
                AABBMax = list(map(int, np.ceil([max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1])])))
                
                # Fill the triangle
                for v in range(AABBMin[1], AABBMax[1]):
                    for u in range(AABBMin[0], AABBMax[0]):
                        if (u < 0 or u >= GL.width) or (v < 0 or v >= GL.height):
                            continue

                        Q = [u, v]
                        P = [p1[0] - p0[0], p1[1] - p0[1]]
                        n = [P[1], -P[0]]
                        QLine = [Q[0] - p0[0], Q[1] - p0[1]]
                        dot = QLine[0] * n[0] + QLine[1] * n[1]
                        if dot < 0:
                            continue

                        P = [p2[0] - p1[0], p2[1] - p1[1]]
                        n = [P[1], -P[0]]
                        QLine = [Q[0] - p1[0], Q[1] - p1[1]]
                        dot = QLine[0] * n[0] + QLine[1] * n[1]
                        if dot < 0:
                            continue

                        P = [p0[0] - p2[0], p0[1] - p2[1]]
                        n = [P[1], -P[0]]
                        QLine = [Q[0] - p2[0], Q[1] - p2[1]]
                        dot = QLine[0] * n[0] + QLine[1] * n[1]
                        if dot < 0:
                            continue
                        
                        # Color interpolation
                        A = p0
                        B = p1
                        C = p2
                        p = [u, v]
                        alpha = (-(p[0] - B[0]) * (C[1] - B[1]) + (p[1] - B[1]) * (C[0] - B[0])) / (-(A[0] - B[0]) * (C[1] - B[1]) + (A[1] - B[1]) * (C[0] - B[0]))
                        beta = (-(p[0] - C[0]) * (A[1] - C[1]) + (p[1] - C[1]) * (A[0] - C[0])) / (-(B[0] - C[0]) * (A[1] - C[1]) + (B[1] - C[1]) * (A[0] - C[0]))
                        gamma = 1 - alpha - beta
                        
                        z = 1 / (alpha / z0 + beta / z1 + gamma / z2)

                        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.depthBuffer) 
                        if (z < gpu.GPU.read_pixel(p, gpu.GPU.DEPTH_COMPONENT32F)):
                            gpu.GPU.draw_pixel(p, gpu.GPU.DEPTH_COMPONENT32F, [z])
                            gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, [z * 255] * 3)

                            _c0 = [c * alpha for c in c0]
                            _c1 = [c * beta for c in c1]
                            _c2 = [c * gamma for c in c2]
                            Cs = [_c0, _c1, _c2]
                            _color = [sum(c * 255) for c in zip(*Cs)]
                            
                            gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer) 
                            gpu.GPU.draw_pixel(p, gpu.GPU.RGB8, _color)
                        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, GL.drawBuffer) 

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        rotationAngle = orientation[3]
        rotationAxis = np.array([orientation[0], orientation[1], orientation[2]])
        rotationAxis = rotationAxis / np.linalg.norm(rotationAxis)

        qx = rotationAxis[0] * np.sin(rotationAngle / 2)
        qy = rotationAxis[1] * np.sin(rotationAngle / 2)
        qz = rotationAxis[2] * np.sin(rotationAngle / 2)
        qw = np.cos(rotationAngle / 2)

        rotationMatrix = np.array([[1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw, 0],
                                   [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz + 2 * qx * qw, 0],
                                   [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2, 0],
                                   [0, 0, 0, 1]])

        translationMatrix = np.array([[1, 0, 0, -position[0]],
                                      [0, 1, 0, -position[1]],
                                      [0, 0, 1, -position[2]],
                                      [0, 0, 0, 1]])

        lookAtMatrix = np.matmul(np.transpose(rotationMatrix), translationMatrix)

        aspectRatio = (GL.width / GL.height)
        fovy = 2 * np.arctan(np.tan(fieldOfView / 2) * GL.height / np.sqrt(GL.height**2 + GL.width**2))
        top = GL.near * np.tan(fovy)
        right = top * aspectRatio

        projectionMatrix = np.array([[GL.near / right, 0, 0, 0],
                                     [0, GL.near / top, 0, 0],
                                     [0, 0, -(GL.far + GL.near) / (GL.far - GL.near), (-2 * GL.far * GL.near) / (GL.far - GL.near)],
                                     [0, 0, -1, 0]])

        # Camera normalized space to pixel space
        screenMatrix = np.array([[GL.width / 2, 0, 0, GL.width / 2],
                                 [0, -GL.height / 2, 0, GL.height / 2],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        cameraMatrix = np.matmul(projectionMatrix, lookAtMatrix) 
        GL.projectionBuffer.append(cameraMatrix)

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.

        translationMatrix = np.array([[1, 0, 0, translation[0]],
                                      [0, 1, 0, translation[1]],
                                      [0, 0, 1, translation[2]],
                                      [0, 0, 0, 1]])

        scaleMatrix = np.array([[scale[0], 0, 0, 0],
                                [0, scale[1], 0, 0],
                                [0, 0, scale[2], 0],
                                [0 ,0, 0, 1]])
        
        rotationAngle = rotation[3]
        rotationAxis = np.array([rotation[0], rotation[1], rotation[2]])
        rotationAxis = rotationAxis / np.linalg.norm(rotationAxis)

        qx = rotationAxis[0] * np.sin(rotationAngle / 2)
        qy = rotationAxis[1] * np.sin(rotationAngle / 2)
        qz = rotationAxis[2] * np.sin(rotationAngle / 2)
        qw = np.cos(rotationAngle / 2)

        rotationMatrix = np.array([[1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), 0],
                                   [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz + qx * qw), 0],
                                   [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2), 0],
                                   [0, 0, 0, 1]])

        modelMatrix = np.matmul(translationMatrix, rotationMatrix)
        modelMatrix = np.matmul(modelMatrix, scaleMatrix)

        parentModelMatrix = GL.transformStack.peek()
        relativeModelMatrix = np.matmul(parentModelMatrix, modelMatrix)
        GL.transformStack.push(relativeModelMatrix);

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        GL.transformStack.pop()

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
