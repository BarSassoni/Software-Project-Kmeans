#ifndef INPUT_H
#define INPUT_H

/* input: fileName; output: points, n, dim */
double **read_points(const char *fileName, int *n, int *dim);
/* input: mat, rows, cols; output: prints matrix */
void print_matrix(double **mat, int rows, int cols);

#endif
