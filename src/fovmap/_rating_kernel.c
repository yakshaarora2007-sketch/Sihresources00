#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    npy_int64 row_min, col_min;
    npy_int64 height, width;
    float *z;
    unsigned char *valid;
} RingMap;

static void free_maps(RingMap *maps) {
    int i;
    for (i = 0; i < 256; ++i) {
        free(maps[i].z);
        free(maps[i].valid);
    }
}

static PyObject *estimate(PyObject *self, PyObject *args) {
    PyObject *rows_obj, *cols_obj, *rings_obj, *z_obj, *labels_obj;
    int radius;
    PyArrayObject *rows = NULL, *cols = NULL, *rings = NULL;
    PyArrayObject *z = NULL, *labels = NULL, *ground = NULL, *has = NULL;
    npy_intp n, i;
    RingMap maps[256];
    npy_int64 min_row[256], max_row[256], min_col[256], max_col[256];
    unsigned char seen[256];
    npy_int64 total_slots = 0;
    int r;
    PyObject *result = NULL;

    memset(maps, 0, sizeof(maps));
    memset(seen, 0, sizeof(seen));
    if (!PyArg_ParseTuple(args, "OOOOOi", &rows_obj, &cols_obj, &rings_obj,
                          &z_obj, &labels_obj, &radius)) {
        return NULL;
    }
    rows = (PyArrayObject *)PyArray_FROM_OTF(rows_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    cols = (PyArrayObject *)PyArray_FROM_OTF(cols_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    rings = (PyArrayObject *)PyArray_FROM_OTF(rings_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    z = (PyArrayObject *)PyArray_FROM_OTF(z_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    labels = (PyArrayObject *)PyArray_FROM_OTF(labels_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (!rows || !cols || !rings || !z || !labels) goto fail;
    n = PyArray_SIZE(rows);
    if (PyArray_SIZE(cols) != n || PyArray_SIZE(rings) != n ||
        PyArray_SIZE(z) != n || PyArray_SIZE(labels) != n) {
        PyErr_SetString(PyExc_ValueError, "kernel input arrays must have equal length");
        goto fail;
    }
    for (i = 0; i < n; ++i) {
        unsigned char ring = *(unsigned char *)PyArray_GETPTR1(rings, i);
        npy_int64 row = *(npy_int32 *)PyArray_GETPTR1(rows, i);
        npy_int64 col = *(npy_int32 *)PyArray_GETPTR1(cols, i);
        if (!seen[ring]) {
            seen[ring] = 1;
            min_row[ring] = max_row[ring] = row;
            min_col[ring] = max_col[ring] = col;
        } else {
            if (row < min_row[ring]) min_row[ring] = row;
            if (row > max_row[ring]) max_row[ring] = row;
            if (col < min_col[ring]) min_col[ring] = col;
            if (col > max_col[ring]) max_col[ring] = col;
        }
    }
    for (r = 0; r < 256; ++r) {
        if (!seen[r]) continue;
        maps[r].row_min = min_row[r];
        maps[r].col_min = min_col[r];
        maps[r].height = max_row[r] - min_row[r] + 1;
        maps[r].width = max_col[r] - min_col[r] + 1;
        if (maps[r].height > 2048 || maps[r].width > 2048) {
            PyErr_Format(PyExc_RuntimeError, "ring %d lattice box exceeds 2048", r);
            goto fail;
        }
        if (maps[r].height > PY_SSIZE_T_MAX / maps[r].width ||
            total_slots > 8000000 - maps[r].height * maps[r].width) {
            PyErr_Format(PyExc_RuntimeError, "ring %d lattice allocation exceeds limit", r);
            goto fail;
        }
        total_slots += maps[r].height * maps[r].width;
        maps[r].z = (float *)malloc((size_t)(maps[r].height * maps[r].width) * sizeof(float));
        maps[r].valid = (unsigned char *)calloc((size_t)(maps[r].height * maps[r].width), 1);
        if (!maps[r].z || !maps[r].valid) {
            PyErr_NoMemory();
            goto fail;
        }
        for (i = 0; i < maps[r].height * maps[r].width; ++i) maps[r].z[i] = NAN;
    }
    ground = (PyArrayObject *)PyArray_EMPTY(1, &n, NPY_FLOAT32, 0);
    has = (PyArrayObject *)PyArray_ZEROS(1, &n, NPY_UINT8, 0);
    if (!ground || !has) goto fail;
    for (i = 0; i < n; ++i) {
        unsigned char ring = *(unsigned char *)PyArray_GETPTR1(rings, i);
        npy_int64 row = *(npy_int32 *)PyArray_GETPTR1(rows, i);
        npy_int64 col = *(npy_int32 *)PyArray_GETPTR1(cols, i);
        if (*(unsigned char *)PyArray_GETPTR1(labels, i) == 1) {
            npy_int64 pos = (row - maps[ring].row_min) * maps[ring].width +
                            (col - maps[ring].col_min);
            maps[ring].z[pos] = *(float *)PyArray_GETPTR1(z, i);
            maps[ring].valid[pos] = 1;
        }
    }
    for (i = 0; i < n; ++i) {
        unsigned char ring = *(unsigned char *)PyArray_GETPTR1(rings, i);
        npy_int64 row = *(npy_int32 *)PyArray_GETPTR1(rows, i);
        npy_int64 col = *(npy_int32 *)PyArray_GETPTR1(cols, i);
        npy_int64 r0 = row - radius, r1 = row + radius;
        npy_int64 c0 = col - radius, c1 = col + radius;
        npy_int64 rr, cc, k = 0, j;
        float values[49];
        if (r0 < maps[ring].row_min) r0 = maps[ring].row_min;
        if (r1 >= maps[ring].row_min + maps[ring].height) r1 = maps[ring].row_min + maps[ring].height - 1;
        if (c0 < maps[ring].col_min) c0 = maps[ring].col_min;
        if (c1 >= maps[ring].col_min + maps[ring].width) c1 = maps[ring].col_min + maps[ring].width - 1;
        for (rr = r0; rr <= r1; ++rr) for (cc = c0; cc <= c1; ++cc) {
            npy_int64 pos = (rr - maps[ring].row_min) * maps[ring].width + (cc - maps[ring].col_min);
            if (maps[ring].valid[pos]) {
                float value = maps[ring].z[pos];
                if (value == value) values[k++] = value;
            }
        }
        {
            int finite = 0;
            for (j = 0; j < k; ++j) if (isfinite(values[j])) { finite = 1; break; }
            if (finite) {
                for (j = 1; j < k; ++j) {
                    float value = values[j];
                    npy_int64 p = j;
                    while (p > 0 && values[p - 1] > value) {
                        values[p] = values[p - 1];
                        --p;
                    }
                    values[p] = value;
                }
                if (k & 1) {
                    *(float *)PyArray_GETPTR1(ground, i) = values[k / 2];
                } else {
                    *(float *)PyArray_GETPTR1(ground, i) =
                        (float)(((double)values[k / 2 - 1] + (double)values[k / 2]) * 0.5);
                }
                *(unsigned char *)PyArray_GETPTR1(has, i) = 1;
            } else if (k > 0) {
                *(float *)PyArray_GETPTR1(ground, i) = values[k / 2];
            }
        }
    }
    result = PyTuple_Pack(2, ground, has);
fail:
    free_maps(maps);
    Py_XDECREF(rows); Py_XDECREF(cols); Py_XDECREF(rings);
    Py_XDECREF(z); Py_XDECREF(labels);
    Py_XDECREF(ground); Py_XDECREF(has);
    return result;
}

static PyMethodDef methods[] = {
    {"estimate", estimate, METH_VARARGS, "Estimate local ground candidates."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_rating_kernel", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__rating_kernel(void) {
    import_array();
    return PyModule_Create(&module);
}
