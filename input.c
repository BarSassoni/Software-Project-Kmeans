#include <ctype.h>
#include <float.h>
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

/* Increase the line buffer while leaving its old pointer valid on error. */
static int grow_line(char **line, size_t *capacity)
{
    char *tmp;
    size_t next;

    if (*capacity > (size_t)-1 / 2) {
        return 0;
    }
    next = *capacity == 0 ? 128 : *capacity * 2;
    tmp = (char *)realloc(*line, next);
    if (tmp == NULL) {
        return 0;
    }
    *line = tmp;
    *capacity = next;
    return 1;
}

/* Read a complete line; return 1 for a line, 0 for EOF, -1 for error. */
static int read_line(FILE *file, char **line, size_t *capacity)
{
    size_t length;
    int c;

    length = 0;
    while ((c = fgetc(file)) != EOF && c != '\n') {
        if (c == '\0') {
            return -1;
        }
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

/* Commas determine the expected number of values on a row. */
static int row_dimension(const char *line)
{
    int dim;

    dim = 1;
    while (*line != '\0') {
        if (*line == ',') {
            if (dim == INT_MAX) {
                return 0;
            }
            dim++;
        }
        line++;
    }
    return dim;
}

/* Parse exactly dim finite values, allowing whitespace around numbers. */
static int parse_row(char *line, double *row, int dim)
{
    char *end;
    int i;
    double value;

    for (i = 0; i < dim; i++) {
        while (isspace((unsigned char)*line)) {
            line++;
        }
        if (*line == '\0') {
            return 0;
        }
        value = strtod(line, &end);
        if (end == line || value != value || value > DBL_MAX || value < -DBL_MAX) {
            return 0;
        }
        row[i] = value;
        while (isspace((unsigned char)*end)) {
            end++;
        }
        if ((i + 1 < dim && *end != ',') || (i + 1 == dim && *end != '\0')) {
            return 0;
        }
        line = end + (i + 1 < dim);
    }
    return 1;
}

/* Make room for one more data point. */
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

/* Empty and whitespace-only lines do not contain data points. */
static int blank_line(const char *line)
{
    while (isspace((unsigned char)*line)) {
        line++;
    }
    return *line == '\0';
}

/* Append a nonblank row after checking its shape and all its values. */
static int append_row(Data *data, char *line)
{
    double *row;
    int dim;

    if (blank_line(line)) {
        return 1;
    }
    dim = row_dimension(line);
    if (dim == 0 || (data->dim != 0 && dim != data->dim) ||
        (size_t)dim > (size_t)-1 / sizeof(double)) {
        return 0;
    }
    row = (double *)malloc((size_t)dim * sizeof(double));
    if (row == NULL) {
        return 0;
    }
    if (!parse_row(line, row, dim) || !grow_points(data)) {
        free(row);
        return 0;
    }
    data->dim = dim;
    data->points[data->n++] = row;
    return 1;
}

/* Read rectangular CSV data, including CRLF and a missing final newline. */
double **read_points(const char *file_name, int *n, int *dim)
{
    FILE *file;
    Data data;
    char *line;
    size_t capacity;
    int status;

    file = fopen(file_name, "r");
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
    if (status < 0 || data.n == 0) {
        free_matrix(data.points, data.n);
        return NULL;
    }
    *n = data.n;
    *dim = data.dim;
    return data.points;
}

/* Print one comma-separated matrix row per line. */
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
