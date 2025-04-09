import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Constants
A, B, C, D = -1.00, 0.16, 1.97, -2.31
WIDTH, HEIGHT = 1980, 1080
MAX_ITER = 2000
ESCAPE_RADIUS = 4.0

@njit
def p(phi):
    return A * phi * phi + B * phi + C

@njit
def T(r, phi):
    return r ** D * np.exp(1j * p(phi))

@njit(parallel=True)
def generate_fractal():
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    scale_x, scale_y = 4.0 / WIDTH, 4.0 / HEIGHT
    
    for y in prange(HEIGHT):
        for x in range(WIDTH):
            total_red = total_green = total_blue = 0
            samples = 4
            offset_x = (x - WIDTH / 2.0) * scale_x
            offset_y = (y - HEIGHT / 2.0) * scale_y
            
            for sy in range(2):
                for sx in range(2):
                    zx = offset_x + (sx + 0.5) * scale_x / 2.0
                    zy = offset_y + (sy + 0.5) * scale_y / 2.0
                    z = complex(zx, zy)
                    c = z
                    
                    for iter in range(MAX_ITER):
                        r, phi = abs(z), np.angle(z)
                        z = T(r, phi) + c
                        if abs(z) > ESCAPE_RADIUS:
                            break
                    
                    if iter == MAX_ITER:
                        red = green = blue = 0
                    else:
                        red = (iter * 15) % 256
                        green = (iter * 5) % 256
                        blue = (iter * 5) % 256
                    
                    total_red += red
                    total_green += green
                    total_blue += blue
            
            image[y, x] = [total_red // samples, total_green // samples, total_blue // samples]
    
    return image

if __name__ == "__main__":
    fractal_image = generate_fractal()
    plt.imsave("fractal.png", fractal_image)
