#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <limits.h>
#include "symnmf.h"

typedef double **(*matrix_function)(double **, int, int);

/* Use the same error message for every failure exposed to Python. */
static PyObject *set_error(PyObject *exception)
{
    PyErr_SetString(exception, "An Error Has Occurred");
    return NULL;
}

/* Get positive matrix dimensions that fit the C interface. */
static int matrix_shape(PyObject *list, int *rows, int *cols)
{
    PyObject *row;
    Py_ssize_t n, dim;

    if (!PyList_Check(list)) {
        set_error(PyExc_ValueError);
        return 0;
    }
    n = PyList_GET_SIZE(list);
    if (n == 0 || n > INT_MAX) {
        set_error(PyExc_ValueError);
        return 0;
    }
    row = PyList_GET_ITEM(list, 0);
    if (!PyList_Check(row)) {
        set_error(PyExc_ValueError);
        return 0;
    }
    dim = PyList_GET_SIZE(row);
    if (dim == 0 || dim > INT_MAX) {
        set_error(PyExc_ValueError);
        return 0;
    }
    *rows = (int)n;
    *cols = (int)dim;
    return 1;
}

/* Copy a rectangular list, rejecting nonfinite or invalid values. */
static int copy_values(PyObject *list, double **mat, int rows,
                       int cols, int nonnegative)
{
    PyObject *row;
    double value;
    int i, j;

    for (i = 0; i < rows; i++) {
        row = PyList_GET_ITEM(list, i);
        if (!PyList_Check(row) || PyList_GET_SIZE(row) != cols) {
            set_error(PyExc_ValueError);
            return 0;
        }
        for (j = 0; j < cols; j++) {
            value = PyFloat_AsDouble(PyList_GET_ITEM(row, j));
            if (PyErr_Occurred() || !(value >= -DBL_MAX &&
                value <= DBL_MAX) || (nonnegative && value < 0.0)) {
                set_error(PyExc_ValueError);
                return 0;
            }
            mat[i][j] = value;
        }
    }
    return 1;
}

/* Allocate a C copy so the caller's Python input stays unchanged. */
static double **read_matrix(PyObject *list, int *rows, int *cols,
                            int nonnegative)
{
    double **mat;

    if (!matrix_shape(list, rows, cols)) {
        return NULL;
    }
    mat = create_matrix(*rows, *cols);
    if (mat == NULL) {
        set_error(PyExc_MemoryError);
        return NULL;
    }
    if (!copy_values(list, mat, *rows, *cols, nonnegative)) {
        free_matrix(mat, *rows);
        return NULL;
    }
    return mat;
}

/* Create one Python row; list insertion takes ownership of each value. */
static PyObject *write_row(double *values, int cols)
{
    PyObject *row, *value;
    int j;

    row = PyList_New(cols);
    if (row == NULL) {
        return set_error(PyExc_MemoryError);
    }
    for (j = 0; j < cols; j++) {
        value = PyFloat_FromDouble(values[j]);
        if (value == NULL) {
            Py_DECREF(row);
            return set_error(PyExc_MemoryError);
        }
        PyList_SET_ITEM(row, j, value);
    }
    return row;
}

/* Convert a result to Python and release its C storage on every path. */
static PyObject *write_matrix(double **mat, int rows, int cols)
{
    PyObject *result, *row;
    int i;

    if (mat == NULL) {
        return set_error(PyExc_RuntimeError);
    }
    result = PyList_New(rows);
    if (result == NULL) {
        free_matrix(mat, rows);
        return set_error(PyExc_MemoryError);
    }
    for (i = 0; i < rows; i++) {
        row = write_row(mat[i], cols);
        if (row == NULL) {
            Py_DECREF(result);
            free_matrix(mat, rows);
            return NULL;
        }
        PyList_SET_ITEM(result, i, row);
    }
    free_matrix(mat, rows);
    return result;
}

/* Shared conversion and cleanup for the three graph matrices. */
static PyObject *calculate(PyObject *args, matrix_function function)
{
    PyObject *list;
    double **points, **result;
    int n, dim;

    if (!PyArg_ParseTuple(args, "O", &list)) {
        return set_error(PyExc_ValueError);
    }
    points = read_matrix(list, &n, &dim, 0);
    if (points == NULL) {
        return NULL;
    }
    result = function(points, n, dim);
    free_matrix(points, n);
    return write_matrix(result, n, n);
}

/* Return the similarity matrix. */
static PyObject *py_sym(PyObject *self, PyObject *args)
{
    (void)self;
    return calculate(args, sym);
}

/* Return the diagonal degree matrix. */
static PyObject *py_ddg(PyObject *self, PyObject *args)
{
    (void)self;
    return calculate(args, ddg);
}

/* Return the normalized similarity matrix. */
static PyObject *py_norm(PyObject *self, PyObject *args)
{
    (void)self;
    return calculate(args, norm);
}

/* Validate both matrices before running the optimization in C. */
static PyObject *optimize(PyObject *hlist, PyObject *wlist)
{
    double **h, **w, **result;
    int n, k, rows, cols;

    h = read_matrix(hlist, &n, &k, 1);
    if (h == NULL) {
        return NULL;
    }
    w = read_matrix(wlist, &rows, &cols, 1);
    if (w == NULL) {
        free_matrix(h, n);
        return NULL;
    }
    if (rows != n || cols != n || k >= n) {
        free_matrix(h, n);
        free_matrix(w, rows);
        return set_error(PyExc_ValueError);
    }
    result = symnmf(h, w, n, k);
    free_matrix(h, n);
    free_matrix(w, rows);
    return write_matrix(result, n, k);
}

/* Optimize the initial H against the supplied normalized matrix W. */
static PyObject *py_symnmf(PyObject *self, PyObject *args)
{
    PyObject *hlist, *wlist;

    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &hlist, &wlist)) {
        return set_error(PyExc_ValueError);
    }
    return optimize(hlist, wlist);
}

static PyMethodDef symnmf_methods[] = {
    {"sym", py_sym, METH_VARARGS, "Return the similarity matrix of points."},
    {"ddg", py_ddg, METH_VARARGS, "Return the diagonal degree matrix."},
    {"norm", py_norm, METH_VARARGS, "Return the normalized similarity matrix."},
    {"symnmf", py_symnmf, METH_VARARGS, "Optimize symNMF from initial H and W."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf_module = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "C calculations for symmetric nonnegative matrix factorization.",
    -1,
    symnmf_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

/* Initialize the Python extension module. */
PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmf_module);
}
