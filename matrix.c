#include <stdlib.h>
#include "symnmf.h"

/* Release the rows and the matrix itself. NULL is allowed. */
void free_matrix(double **mat, int rows)
{
    int i;

    if (mat == NULL) {
        return;
    }
    for (i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

/* Create a zero-filled matrix, cleaning up on allocation failure. */
double **create_matrix(int rows, int cols)
{
    double **mat;
    int i;

    if (rows <= 0 || cols <= 0 ||
        (size_t)rows > (size_t)-1 / sizeof(double *) ||
        (size_t)cols > (size_t)-1 / sizeof(double)) {
        return NULL;
    }
    mat = (double **)calloc((size_t)rows, sizeof(double *));
    if (mat == NULL) {
        return NULL;
    }
    for (i = 0; i < rows; i++) {
        mat[i] = (double *)calloc((size_t)cols, sizeof(double));
        if (mat[i] == NULL) {
            free_matrix(mat, i);
            return NULL;
        }
    }
    return mat;
}
