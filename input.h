#ifndef INPUT_H
#define INPUT_H

double **read_points(const char *file_name, int *n, int *dim);
void print_matrix(double **mat, int rows, int cols);

#endif
