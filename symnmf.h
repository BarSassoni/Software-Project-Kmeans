#ifndef SYMNMF_H
#define SYMNMF_H

double **create_matrix(int rows, int cols);
void free_matrix(double **mat, int rows);
double **sym(double **points, int n, int dim);
double **ddg(double **points, int n, int dim);
double **norm(double **points, int n, int dim);
double **symnmf(double **h, double **w, int n, int k);

#endif
