#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Renderizador X3D.

Desenvolvido por: Luciano Soares <lpsoares@insper.edu.br>
Disciplina: Computação Gráfica
Data: 28 de Agosto de 2020
"""

import os           # Para rotinas do sistema operacional
import argparse     # Para tratar os parâmetros da linha de comando

import gl           # Recupera rotinas de suporte ao X3D

import interface    # Janela de visualização baseada no Matplotlib
import gpu          # Simula os recursos de uma GPU

import x3d          # Faz a leitura do arquivo X3D, gera o grafo de cena e faz traversal
import scenegraph   # Imprime o grafo de cena no console

LARGURA = 60  # Valor padrão para largura da tela
ALTURA = 40   # Valor padrão para altura da tela


class Renderizador:
    """Realiza a renderização da cena informada."""

    def __init__(self):
        """Definindo valores padrão."""
        self.width = LARGURA
        self.height = ALTURA
        self.x3d_file = ""
        self.image_file = "tela.png"
        self.scene = None
        self.framebuffers = {}
        self.supersampling = False
        self.samplingLevel = 2

    def setup(self):
        """Configura o sistema para a renderização."""
        # Configurando color buffers para exibição na tela

        # Cria uma (1) posição de FrameBuffer na GPU
        fbo = gpu.GPU.gen_framebuffers(4)

        # Define o atributo FRONT como o FrameBuffe principal
        self.framebuffers["FRONT"] = fbo[0]
        self.framebuffers["BACK"] = fbo[1]
        self.framebuffers["SSAA"] = fbo[2]
        self.framebuffers["DEPTH"] = fbo[3]

        # Opções:
        # - DRAW_FRAMEBUFFER: Faz o bind só para escrever no framebuffer
        # - READ_FRAMEBUFFER: Faz o bind só para leitura no framebuffer
        # - FRAMEBUFFER: Faz o bind para leitura e escrita no framebuffer

        # Aloca memória no FrameBuffer para um tipo e tamanho especificado de buffer

        # Memória de Framebuffer para canal de cores
        gpu.GPU.framebuffer_storage(
            self.framebuffers["FRONT"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width,
            self.height
        )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["BACK"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width,
            self.height
        )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["DEPTH"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width * self.samplingLevel if self.supersampling else self.width,
            self.height * self.samplingLevel if self.supersampling else self.height
        )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["DEPTH"],
            gpu.GPU.DEPTH_ATTACHMENT,
            gpu.GPU.DEPTH_COMPONENT32F,
            self.width * self.samplingLevel if self.supersampling else self.width,
            self.height * self.samplingLevel if self.supersampling else self.height
        )

        gpu.GPU.framebuffer_storage(
            self.framebuffers["SSAA"],
            gpu.GPU.COLOR_ATTACHMENT,
            gpu.GPU.RGB8,
            self.width * self.samplingLevel if self.supersampling else self.width,
            self.height * self.samplingLevel if self.supersampling else self.height
        )

        if (self.supersampling):
            gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["SSAA"])
        else:
            gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["FRONT"])

        # Opções:
        # - COLOR_ATTACHMENT: alocações para as cores da imagem renderizada
        # - DEPTH_ATTACHMENT: alocações para as profundidades da imagem renderizada
        # Obs: Você pode chamar duas vezes a rotina com cada tipo de buffer.

        # Tipos de dados:
        # - RGB8: Para canais de cores (Vermelho, Verde, Azul) 8bits cada (0-255)
        # - RGBA8: Para canais de cores (Vermelho, Verde, Azul, Transparência) 8bits cada (0-255)
        # - DEPTH_COMPONENT16: Para canal de Profundidade de 16bits (half-precision) (0-65535)
        # - DEPTH_COMPONENT32F: Para canal de Profundidade de 32bits (single-precision) (float)

        # Define cor que ira apagar o FrameBuffer quando clear_buffer() invocado
        gpu.GPU.clear_color([0, 0, 0])

        # Define a profundidade que ira apagar o FrameBuffer quando clear_buffer() invocado
        # Assuma 1.0 o mais afastado e -1.0 o mais próximo da camera
        gpu.GPU.clear_depth(1.0)

        # Definindo tamanho do Viewport para renderização
        self.scene.viewport(width=self.width, height=self.height)

        # Fill z buffer with infinite values
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["DEPTH"])
        for j in range(self.height):
            for i in range(self.width):
                gpu.GPU.draw_pixel([i, j], gpu.GPU.DEPTH_COMPONENT32F, [float("inf")])
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["FRONT"])

    def pre(self):
        """Rotinas pré renderização."""
        # Função invocada antes do processo de renderização iniciar.

        # Limpa o frame buffers atual
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["FRONT"])
        gpu.GPU.clear_buffer()
        gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["DEPTH"])
        gpu.GPU.clear_buffer()

        if (self.supersampling):
           gpu.GPU.bind_framebuffer(gpu.GPU.FRAMEBUFFER, self.framebuffers["SSAA"])
        
        # Recursos que podem ser úteis:
        # Define o valor do pixel no framebuffer: draw_pixel(coord, mode, data)
        # Retorna o valor do pixel no framebuffer: read_pixel(coord, mode)

    def pos(self):
        """Rotinas pós renderização."""
        # Função invocada após o processo de renderização terminar.

        if (self.supersampling):
            gpu.GPU.bind_framebuffer(gpu.GPU.DRAW_FRAMEBUFFER, self.framebuffers["FRONT"])

            for j in range(0, self.height * self.samplingLevel, self.samplingLevel):
                for i in range(0, self.width * self.samplingLevel, self.samplingLevel):
                    colors = [] 
                    meanFactor = 1.0 / self.samplingLevel**2
                    for v in range(self.samplingLevel):
                        for u in range(self.samplingLevel):
                            colors.append(gpu.GPU.read_pixel([i + u, j + v], gpu.GPU.RGB8) * meanFactor)
                    averageColor = [sum(c) for c in zip(*colors)]
                    gpu.GPU.draw_pixel([i // self.samplingLevel, j // self.samplingLevel], gpu.GPU.RGB8, averageColor)

        # Método para a troca dos buffers (NÃO IMPLEMENTADO)
        gpu.GPU.swap_buffers()

        gl.GL.lights = []

    def mapping(self):
        """Mapeamento de funções para as rotinas de renderização."""
        # Rotinas encapsuladas na classe GL (Graphics Library)
        x3d.X3D.renderer["Polypoint2D"] = gl.GL.polypoint2D
        x3d.X3D.renderer["Polyline2D"] = gl.GL.polyline2D
        x3d.X3D.renderer["TriangleSet2D"] = gl.GL.triangleSet2D
        x3d.X3D.renderer["TriangleSet"] = gl.GL.triangleSet
        x3d.X3D.renderer["Viewpoint"] = gl.GL.viewpoint
        x3d.X3D.renderer["Transform_in"] = gl.GL.transform_in
        x3d.X3D.renderer["Transform_out"] = gl.GL.transform_out
        x3d.X3D.renderer["TriangleStripSet"] = gl.GL.triangleStripSet
        x3d.X3D.renderer["IndexedTriangleStripSet"] = gl.GL.indexedTriangleStripSet
        x3d.X3D.renderer["IndexedFaceSet"] = gl.GL.indexedFaceSet
        x3d.X3D.renderer["Box"] = gl.GL.box
        x3d.X3D.renderer["Sphere"] = gl.GL.sphere
        x3d.X3D.renderer["NavigationInfo"] = gl.GL.navigationInfo
        x3d.X3D.renderer["DirectionalLight"] = gl.GL.directionalLight
        x3d.X3D.renderer["PointLight"] = gl.GL.pointLight
        x3d.X3D.renderer["Fog"] = gl.GL.fog
        x3d.X3D.renderer["TimeSensor"] = gl.GL.timeSensor
        x3d.X3D.renderer["SplinePositionInterpolator"] = gl.GL.splinePositionInterpolator
        x3d.X3D.renderer["OrientationInterpolator"] = gl.GL.orientationInterpolator

    def render(self):
        """Laço principal de renderização."""
        self.pre()  # executa rotina pré renderização
        self.scene.render()  # faz o traversal no grafo de cena
        self.pos()  # executa rotina pós renderização
        return gpu.GPU.get_frame_buffer()

    def main(self):
        """Executa a renderização."""
        # Tratando entrada de parâmetro
        parser = argparse.ArgumentParser(add_help=False)   # parser para linha de comando
        parser.add_argument("-i", "--input", help="arquivo X3D de entrada")
        parser.add_argument("-o", "--output", help="arquivo 2D de saída (imagem)")
        parser.add_argument("-w", "--width", help="resolução horizonta", type=int)
        parser.add_argument("-h", "--height", help="resolução vertical", type=int)
        parser.add_argument("-g", "--graph", help="imprime o grafo de cena", action='store_true')
        parser.add_argument("-p", "--pause", help="começa simulação em pausa", action='store_true')
        parser.add_argument("-q", "--quiet", help="não exibe janela", action='store_true')
        args = parser.parse_args() # parse the arguments
        if args.input:
            self.x3d_file = args.input
        if args.output:
            self.image_file = args.output
        if args.width:
            self.width = args.width
        if args.height:
            self.height = args.height

        path = os.path.dirname(os.path.abspath(self.x3d_file))

        # Iniciando simulação de GPU
        gpu.GPU(self.image_file, path)

        # Abre arquivo X3D
        self.scene = x3d.X3D(self.x3d_file)

        # Funções que irão fazer o rendering
        self.mapping()

        # Se no modo silencioso não configurar janela de visualização
        if not args.quiet:
            window = interface.Interface(self.width, self.height, self.x3d_file)
            self.scene.set_preview(window)

        # carrega os dados do grafo de cena
        if self.scene:
            self.scene.parse()
            if args.graph:
                scenegraph.Graph(self.scene.root)

        # Configura o sistema para a renderização.
        self.setup()

        # Iniciando Biblioteca Gráfica
        gl.GL.setup(
            self.width * self.samplingLevel if self.supersampling else self.width,
            self.height * self.samplingLevel if self.supersampling else self.height,
            self.framebuffers["SSAA"] if self.supersampling else self.framebuffers["FRONT"],
            self.framebuffers["DEPTH"],
            self.x3d_file.rsplit('/', 1)[0] ,
            near=0.01,
            far=1000
        )

        # Se no modo silencioso salvar imagem e não mostrar janela de visualização
        if args.quiet:
            gpu.GPU.save_image()  # Salva imagem em arquivo
        else:
            window.set_saver(gpu.GPU.save_image)  # pasa a função para salvar imagens
            window.preview(args.pause, self.render)  # mostra visualização

if __name__ == '__main__':
    renderizador = Renderizador()
    renderizador.main()
