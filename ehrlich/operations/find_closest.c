#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static PyObject* find(PyObject* self, PyObject* args) {
    PyArrayObject *A = NULL, *B = NULL;
    
    // Parse input objects (two numpy arrays)
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B)) {
        return NULL;
    }
    
    // Ensure both A and B are 2D arrays with shape (N, 3)
    if (PyArray_NDIM(A) != 2 || PyArray_NDIM(B) != 2) {
        PyErr_SetString(PyExc_ValueError, "Both A and B must be 2D matrices");
        return NULL;
    }
    
    npy_intp *dims_A = PyArray_DIMS(A);
    npy_intp *dims_B = PyArray_DIMS(B);

    // Check if both arrays have exactly 3 columns (for 3D points)
    if (dims_A[1] != 3 || dims_B[1] != 3) {
        PyErr_SetString(PyExc_ValueError, "Both A and B must have exactly 3 columns");
        return NULL;
    }

    npy_intp num_A = dims_A[0]; // Number of points in A
    npy_intp num_B = dims_B[0]; // Number of points in B

    // Create output arrays for indices (I) and distances (D)
    npy_intp I_dims[1] = {num_A};
    npy_intp D_dims[1] = {num_A};
    
    PyArrayObject *I = (PyArrayObject *) PyArray_SimpleNew(1, I_dims, NPY_INT32);
    PyArrayObject *D = (PyArrayObject *) PyArray_SimpleNew(1, D_dims, NPY_FLOAT32);
    
    // Access raw data pointers
    float *A_data = (float *) PyArray_DATA(A);
    float *B_data = (float *) PyArray_DATA(B);
    int *I_data = (int *) PyArray_DATA(I);
    float *D_data = (float *) PyArray_DATA(D);

    // For each point in A, find the nearest point in B
    for (npy_intp i = 0; i < num_A; i++) {
        float min_dist_sq = INFINITY;
        int min_index = -1;

        // Calculate distance from A[i] to all points in B
        for (npy_intp j = 0; j < num_B; j++) {
            float dist_sq = 0.0;

            // Compute squared distance (for 3D points)
            for (npy_intp k = 0; k < 3; k++) {
                float diff = A_data[i * 3 + k] - B_data[j * 3 + k];
                dist_sq += diff * diff;
            }

            // Update if this distance is the smallest
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                min_index = j;
            }
        }

        // Store the closest index and distance (take square root of the distance)
        I_data[i] = min_index;
        D_data[i] = sqrtf(min_dist_sq);
    }

    // Return tuple (I, D)
    PyObject *result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, (PyObject *) I);
    PyTuple_SET_ITEM(result, 1, (PyObject *) D);

    return result;
}


static PyMethodDef methods[] = {
    {"find",  find, METH_VARARGS, "Find closeses points"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef find_closest = {
    PyModuleDef_HEAD_INIT,
    "find_closest",
    "This is a module name abc123",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_find_closest() {
    PyObject *module = PyModule_Create(&find_closest);
    import_array();
    return module;
}
