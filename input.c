#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "input.h"
#include "symnmf.h"

typedef struct {
    double **points;
    int n;
    int dim;
    int capacity;
} Data;

/* increase linebuffer */
/* input: line, capacity; output: success flag */
static int grow_line(char **line, size_t *capacity)
{
    char *tmp;
    size_t next;
    if (*capacity > (size_t)-1 / 2) {
        return 0;
    }
    next = *capacity == 0 ? 128 : *capacity * 2;
    tmp = (char *)realloc(*line, next);
    /*tmp אם keep original memory if realloc fail */
    if (tmp == NULL) {
        return 0;
    }
    *line = tmp;
    *capacity = next;
    return 1;
}

/* read line, 1 for line, 0 for EOF, -1 for err */
/* input: file, line, capacity; output: line status */
static int read_line(FILE *file, char **line, size_t *capacity)
{
    size_t length;
    int c;
    length = 0;
    while ((c = fgetc(file)) != EOF && c != '\n') {
        if (length + 1 >= *capacity && !grow_line(line, capacity)) {
            return -1;
        }
        (*line)[length++] = (char)c;
    }
    if (ferror(file)) {
        return -1;
    }
    if (c == EOF && length == 0) {
        return 0;
    }
    if (length + 1 >= *capacity && !grow_line(line, capacity)) {
        return -1;
    }
    (*line)[length] = '\0';
    return 1;
}

/* commas =num of vals on row */
/* input: line; output: dimension */
static int row_dimension(const char *line)
{
    int dim;
    dim = 1;
    while (*line != '\0') {
        if (*line == ',') {
            dim++;
        }
        line++;
    }
    return dim;
}

/* convert the row's values to doubles */
/* input: line, row, dim; output: fills row */
static void parse_row(char *line, double *row, int dim)
{
    int i;
    for (i = 0; i < dim; i++) {
        row[i] = strtod(line, &line);
        if (i + 1 < dim) {
            line++;
        }
    }
}

/* space for data point */
/* input: data; output: success flag */
static int grow_points(Data *data)
{
    double **tmp;
    int capacity;

    if (data->n < data->capacity) {
        return 1;
    }
    if (data->capacity > INT_MAX / 2) {
        return 0;
    }
    capacity = data->capacity == 0 ? 16 : data->capacity * 2;
    if ((size_t)capacity > (size_t)-1 / sizeof(double *)) {
        return 0;
    }
    tmp = (double **)realloc(data->points, (size_t)capacity * sizeof(double *));
    if (tmp == NULL) {
        return 0;
    }
    data->points = tmp;
    data->capacity = capacity;
    return 1;
}

/* allocate and add a data point */
/* input: data, line; output: success flag */
static int append_row(Data *data, char *line)
{
    double *row;
    int dim;

    dim = row_dimension(line);
    if ((size_t)dim > (size_t)-1 / sizeof(double)) {
        return 0;
    }
    row = (double *)malloc((size_t)dim * sizeof(double));
    if (row == NULL) {
        return 0;
    }
    parse_row(line, row, dim);
    if (!grow_points(data)) {
        free(row);
        return 0;
    }
    data->dim = dim;
    data->points[data->n] = row;
    data->n++;
    return 1;
}

/* read file data (CRLF and a missing final newline ) */
/* input: fileName; output: points, n, dim */
double **read_points(const char *fileName, int *n, int *dim)
{
    FILE *file;
    Data data;
    char *line;
    size_t capacity;
    int status;
    file = fopen(fileName, "r");
    if (file == NULL) {
        return NULL;
    }
    data.points = NULL;
    data.n = 0;
    data.dim = 0;
    data.capacity = 0;
    line = NULL;
    capacity = 0;
    while ((status = read_line(file, &line, &capacity)) == 1) {
        if (!append_row(&data, line)) {
            status = -1;
            break;
        }
    }
    free(line);
    if (fclose(file) != 0) {
        status = -1;
    }
    if (status < 0) {
        free_matrix(data.points, data.n);
        return NULL;
    }
    *n = data.n;
    *dim = data.dim;
    return data.points;
}

/* print matrix row per line (with , ) */
/* input: mat, rows, cols; output: prints matrix */
void print_matrix(double **mat, int rows, int cols)
{
    int i;
    int j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (j > 0) {
                printf(",");
            }
            printf("%.4f", mat[i][j]);
        }
        printf("\n");
    }
}
