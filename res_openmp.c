#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <jpeglib.h>
#include <omp.h>

typedef struct {
    unsigned char *data;
    int width;
    int height;
} Image;

Image *readJPEG(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        return NULL;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    Image *image = (Image *)malloc(sizeof(Image));
    if (!image) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    image->width = cinfo.output_width;
    image->height = cinfo.output_height;
    image->data = (unsigned char *)malloc(cinfo.output_width * cinfo.output_height * cinfo.output_components * sizeof(unsigned char));
    if (!image->data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        jpeg_destroy_decompress(&cinfo);
        free(image);
        return NULL;
    }

    unsigned char *row_pointer = image->data;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
        row_pointer += cinfo.output_width * cinfo.output_components;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);

    return image;
}

void writeJPEG(const char *filename, Image *image) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Unable to create file %s\n", filename);
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = image->width;
    cinfo.image_height = image->height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    unsigned char *row_pointer = image->data;
    while (cinfo.next_scanline < cinfo.image_height) {
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
        row_pointer += cinfo.image_width * cinfo.input_components;
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(file);
}

Image *downscale(Image *input, int factor) {
    int new_width = input->width / factor;
    int new_height = input->height / factor;

    Image *output = (Image *)malloc(sizeof(Image));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    output->width = new_width;
    output->height = new_height;
    output->data = (unsigned char *)malloc(new_width * new_height * 3 * sizeof(unsigned char));
    if (!output->data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(output);
        return NULL;
    }
    omp_set_num_threads(20);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < new_height; ++i) {
        for (int j = 0; j < new_width; ++j) {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            for (int k = 0; k < factor; ++k) {
                for (int l = 0; l < factor; ++l) {
                    sum_r += input->data[((i * factor + k) * input->width + (j * factor + l)) * 3];
                    sum_g += input->data[((i * factor + k) * input->width + (j * factor + l)) * 3 + 1];
                    sum_b += input->data[((i * factor + k) * input->width + (j * factor + l)) * 3 + 2];
                }
            }
            output->data[(i * new_width + j) * 3] = sum_r / (factor * factor);
            output->data[(i * new_width + j) * 3 + 1] = sum_g / (factor * factor);
            output->data[(i * new_width + j) * 3 + 2] = sum_b / (factor * factor);
        }
    }

    return output;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_folder> <output_folder> <scale_factor>\n", argv[0]);
        return 1;
    }

    double start_time = omp_get_wtime();

    const char *input_folder = argv[1];
    const char *output_folder = argv[2];
    int scale_factor = atoi(argv[3]);

    if (scale_factor <= 0) {
        fprintf(stderr, "Error: Invalid scale factor\n");
        return 1;
    }

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_folder)) != NULL) {
        #pragma omp parallel private(ent)
        {
            while (1) {
                #pragma omp critical
                ent = readdir(dir);

                if (ent == NULL) break;

                if (ent->d_type == DT_REG) { // Regular file
                    char input_path[257];
                    snprintf(input_path, sizeof(input_path), "%s/%s", input_folder, ent->d_name);

                    Image *input_image = readJPEG(input_path);
                    if (!input_image) {
                        continue;
                    }

                    Image *output_image = downscale(input_image, scale_factor);
                    if (!output_image) {
                        free(input_image->data);
                        free(input_image);
                        continue;
                    }

                    char output_path[257];
                    snprintf(output_path, sizeof(output_path), "%s/%s", output_folder, ent->d_name);

                    writeJPEG(output_path, output_image);

                    free(input_image->data);
                    free(input_image);
                    free(output_image->data);
                    free(output_image);
                }
            }
        }
        closedir(dir);
    } else {
        perror("Error opening directory");
        return 1;
    }

    double end_time = omp_get_wtime();
    printf("Total execution time: %.2f seconds\n", end_time - start_time);

    return 0;
}