#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "symnmf.h"

#ifndef SYMNMF_EXTENSION
#include <stdio.h>
#include <string.h>
#include "input.h"
#endif

#define EPSILON 0.0001
#define MAX_ITER 300
#define BETA 0.5

/* symmetric similarity matrix 0's diag */
/* input: points, n, dim; output: similarity matrix */
double **sym(double **points, int n, int dim)
{
    double **a;
    double sum;
    double diff;
    int i;
    int j;
    int d;
    if (points == NULL || n <= 0 || dim <= 0) {
        return NULL;
    }
    a = create_matrix(n, n);
    if (a == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            sum = 0.0;
            for (d = 0; d < dim; d++) {
                diff = points[i][d] - points[j][d];
                sum += diff * diff;
            }
            a[i][j] = exp(-sum / 2.0);
            a[j][i] = a[i][j];
        }
    }
    return a;
}

/*sum rows to find degree */
/* input: a, n; output: row degrees */
static double *row_degrees(double **a, int n)
{
    double *degrees;
    int i;
    int j;

    degrees = (double *)calloc((size_t)n, sizeof(double));
    if (degrees == NULL) {
        return NULL;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            degrees[i] += a[i][j];
        }
    }
    return degrees;
}

/* return diagonal matrix with similarity row sums */
/* input: points, n, dim; output: degree matrix */
double **ddg(double **points, int n, int dim)
{
    double **d;
    double degree;
    int i;
    int j;

    d = sym(points, n, dim);
    if (d == NULL) {
        return NULL;
    }
    /* similarity row after entries sum */
    for (i = 0; i < n; i++) {
        degree = 0.0;
        for (j = 0; j < n; j++) {
            degree += d[i][j];
            d[i][j] = 0.0;
        }
        d[i][i] = degree;
    }
    return d;
}

/* zero degrees, leave zero row or column */
/* input: points, n, dim; output: normalized matrix */
double **norm(double **points, int n, int dim)
{
    double **a;
    double *degrees;
    int i;
    int j;
    a = sym(points, n, dim);
    if (a == NULL) {
        return NULL;
    }
    degrees = row_degrees(a, n);
    if (degrees == NULL) {
        free_matrix(a, n);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        degrees[i] = sqrt(degrees[i]);
    }
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            if (degrees[i] > 0.0 && degrees[j] > 0.0) {
                a[i][j] = a[i][j] / degrees[i] / degrees[j];
            }
            a[j][i] = a[i][j];
        }
    }
    free(degrees);
    return a;
}

/* calculation H^t * H */
/* input: h, gram, n, k; output: fills gram */
static void gram_matrix(double **h, double **gram, int n, int k)
{
    int i;
    int j;
    int row;
    double sum;

    for (i = 0; i < k; i++) {
        for (j = i; j < k; j++) {
            sum = 0.0;
            for (row = 0; row < n; row++) {
                sum += h[row][i] * h[row][j];
            }
            gram[i][j] = sum;
            gram[j][i] = sum;
        }
    }
}

/*multiply row with col */
/* input: row, mat, col, length; output: dot product */
static double row_product(double *row, double **mat, int col, int length)
{
    double sum;
    int i;

    sum = 0.0;
    for (i = 0; i < length; i++) {
        sum += row[i] * mat[i][col];
    }
    return sum;
}

/* simultaneou update and zeros stay zeros */
/* input: h, next, w, gram, n, k, change; output: success, next, change */
static int update_h(double **h, double **next, double **w, double **gram, int n, int k, double *change)
{
    int i;
    int j;
    double numerator;
    double denominator;
    double value;
    double diff;

    *change = 0.0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            numerator = row_product(w[i], h, j, n);
            denominator = row_product(h[i], gram, j, k);
            value = 0.0;
            if (h[i][j] != 0.0) {
                if (denominator <= 0.0) {
                    return 0;
                }
                value = h[i][j] * (1.0 - BETA + BETA * numerator / denominator);
            }
            if (value != value || value > DBL_MAX || value < 0.0) {
                return 0;
            }
            next[i][j] = value;
            diff = value - h[i][j];
            *change += diff * diff;
        }
    }
    return *change <= DBL_MAX;
}

/*copy values to target matrix */
/* input: source, target, n, k; output: fills target */
static void copy_matrix(double **source, double **target, int n, int k)
{
    int i;
    int j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            target[i][j] = source[i][j];
        }
    }
}

/* return new H */
/* input: h, w, n, k; output: new H */
double **symnmf(double **h, double **w, int n, int k)
{
    double **current;
    double **next;
    double **gram;
    double **tmp;
    double change;
    int iter;

    current = create_matrix(n, k);
    next = create_matrix(n, k);
    gram = create_matrix(k, k);
    if (h == NULL || w == NULL || current == NULL || next == NULL || gram == NULL) {
        free_matrix(current, n);
        free_matrix(next, n);
        free_matrix(gram, k);
        return NULL;
    }
    copy_matrix(h, current, n, k);
    for (iter = 0; iter < MAX_ITER; iter++) {
        /* calculate H^T H for the update step */
        gram_matrix(current, gram, n, k);
        if (!update_h(current, next, w, gram, n, k, &change)) {
            free_matrix(current, n);
            current = NULL;
            break;
        }
        /* use update H and old matrix in next iterr */
        tmp = current;
        current = next;
        next = tmp;
        if (change < EPSILON) {
            break;
        }
    }
    free_matrix(next, n);
    free_matrix(gram, k);
    return current;
}

#ifndef SYMNMF_EXTENSION
/* select matrix calc */
/* input: goal, points, n, dim; output: goal matrix */
static double **calculate_goal(char *goal, double **points, int n, int dim)
{
    if (strcmp(goal, "sym") == 0) {
        return sym(points, n, dim);
    }
    if (strcmp(goal, "ddg") == 0) {
        return ddg(points, n, dim);
    }
    if (strcmp(goal, "norm") == 0) {
        return norm(points, n, dim);
    }
    return NULL;
}

/* read data, calculate goal and print 4 decimal places */
/* input: argc, argv; output: exit status */
int main(int argc, char *argv[])
{
    double **points;
    double **result;
    int n;
    int dim;

    if (argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    points = read_points(argv[2], &n, &dim);
    if (points == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    result = calculate_goal(argv[1], points, n, dim);
    free_matrix(points, n);
    if (result == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    print_matrix(result, n, n);
    free_matrix(result, n);
    return 0;
}
#endif
