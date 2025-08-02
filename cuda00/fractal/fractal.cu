#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <png.h>
#include <omp.h>

#define A -1.00
#define B 0.16
#define C 1.97
#define D -2.31

#define WIDTH 1980
#define HEIGHT 1080
#define MAX_ITER 2000
#define ESCAPE_RADIUS 4.0

double p(double phi) {
    return A * phi * phi + B * phi + C;
}

double complex T(double r, double phi) {
    return cpow(r, D) * cexp(I * p(phi));
}

void generate_fractal(png_bytepp row_pointers) {
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < HEIGHT; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < WIDTH; x++) {
            double total_red = 0, total_green = 0, total_blue = 0;
            int samples = 4;

            double scale_x = 4.0 / WIDTH;
            double scale_y = 4.0 / HEIGHT;
            double offset_x = (x - WIDTH / 2.0) * scale_x;
            double offset_y = (y - HEIGHT / 2.0) * scale_y;

            for (int sy = 0; sy < 2; sy++) {
                for (int sx = 0; sx < 2; sx++) {
                    double zx = offset_x + (sx + 0.5) * scale_x / 2.0;
                    double zy = offset_y + (sy + 0.5) * scale_y / 2.0;
                    double complex z = zx + zy * I;
                    double complex c = z;

                    int iter;
                    for (iter = 0; iter < MAX_ITER; iter++) {
                        double r = cabs(z);
                        double phi = carg(z);
                        z = T(r, phi) + c;
                        if (cabs(z) > ESCAPE_RADIUS) break;
                    }

                    png_byte red, green, blue;
                    if (iter == MAX_ITER) {
                        red = green = blue = 0;
                    } else {
                        red = (iter * 15) % 256;
                        green = (iter * 5) % 256;
                        blue = (iter * 5) % 256;
                    }

                    total_red += red;
                    total_green += green;
                    total_blue += blue;
                }
            }

            row[x * 3] = total_red / samples;
            row[x * 3 + 1] = total_green / samples;
            row[x * 3 + 2] = total_blue / samples;
        }
    }
}
