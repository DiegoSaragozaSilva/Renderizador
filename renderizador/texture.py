from Math.vec2 import Vec2
from Math.vec3 import Vec3

from PIL import Image
from math import sqrt, log2

class Texture:
    def __init__(self, path: str = None, mipmap: bool = False):
        self.dimensions = Vec2()
        self.image = None
        self.mipmaps = list()
        if path:
            self.load_from_path(path)

            if mipmap:
                new_dimensions = self.dimensions / 2.0
                new_image = self.image.resize((int(new_dimensions.x), int(new_dimensions.y)))

                new_texture = Texture()
                new_texture.dimensions = new_dimensions
                new_texture.image = new_image
                self.mipmaps.append(new_texture)

                while new_dimensions.x > 2.0 and new_dimensions.y > 2.0:
                    new_dimensions /= 2.0
                    new_image = new_image.resize((int(new_dimensions.x), int(new_dimensions.y)))
                    
                    new_texture = Texture()
                    new_texture.dimensions = new_dimensions
                    new_texture.image = new_image
                    self.mipmaps.append(new_texture)

    def load_from_path(self, path):
        self.image = Image.open(path).convert("RGB")
        self.dimensions = Vec2(self.image.size[0], self.image.size[1])

    def read_pixel(self, uv: Vec2) -> Vec2:
        pixel_coord = Vec2(uv.x * self.dimensions.x, (1.0 - uv.y) * self.dimensions.y)

        uv01 = Vec2(pixel_coord.x, pixel_coord.y + 1)
        uv10 = Vec2(pixel_coord.x + 1, pixel_coord.y)
        a = self.get_mipmap_level(pixel_coord, uv01, uv10)

        r, g, b = self.image.getpixel((pixel_coord.x, pixel_coord.y))
        return Vec3(r, g, b)

    def get_mipmap_level(self, uv00: Vec2, uv01: Vec2, uv10 : Vec2) -> int:
        dudx = uv10.x - uv00.x
        dvdx = uv10.y - uv00.y
        
        dudy = uv01.x - uv00.x
        dvdy = uv01.y - uv00.y

        L = max(sqrt(dudx**2 + dvdx**2), sqrt(dudy**2 + dvdy**2))
        D = log2(L)
