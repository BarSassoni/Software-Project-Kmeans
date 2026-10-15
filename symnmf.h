#ifndef SYMNMF_H
#define SYMNMF_H

/* input: rows, cols; output: zero matrix */
double **create_matrix(int rows, int cols);
/* input: mat, rows; output: frees matrix */
void free_matrix(double **mat, int rows);
/* input: points, n, dim; output: similarity matrix */
double **sym(double **points, int n, int dim);
/* input: points, n, dim; output: degree matrix */
double **ddg(double **points, int n, int dim);
/* input: points, n, dim; output: normalized matrix */
double **norm(double **points, int n, int dim);
/* input: h, w, n, k; output: updated H matrix */
double **symnmf(double **h, double **w, int n, int k);

#endif
