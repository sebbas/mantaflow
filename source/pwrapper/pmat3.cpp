/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2024 Sebastian Barschkis, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Matrix3x3 and Matrix2x2 class extension for python
 *
 ******************************************************************************/

#include "pythonInclude.h"
#include <string>
#include <sstream>
#include "vectorbase.h"
#include "structmember.h"
#include "manta.h"
#include "matrixbase.h"

using namespace std;

namespace Manta {

extern PyTypeObject PbMat3Type;

struct PbMat3 {
	PyObject_HEAD
	float data[9];
};

static PyMethodDef PbMat3Methods[] = {
	{NULL}  // Sentinel
};

static PyMemberDef PbMat3Members[] = {
	{(char*)"v00", T_FLOAT, offsetof(PbMat3, data), 0, (char*)"v00"},
	{(char*)"v01", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*1, 0, (char*)"v01"},
	{(char*)"v02", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*2, 0, (char*)"v02"},
	{(char*)"v10", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*3, 0, (char*)"v10"},
	{(char*)"v11", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*4, 0, (char*)"v11"},
	{(char*)"v12", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*5, 0, (char*)"v12"},
	{(char*)"v20", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*6, 0, (char*)"v20"},
	{(char*)"v21", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*7, 0, (char*)"v21"},
	{(char*)"v22", T_FLOAT, offsetof(PbMat3, data)+sizeof(float)*8, 0, (char*)"v22"},
	{NULL}  // Sentinel
};

static void PbMat3Dealloc(PbMat3* self) {
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * PbMat3New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
	return type->tp_alloc(type, 0);
}

static int PbMat3Init(PbMat3 *self, PyObject *args, PyObject *kwds) {
	
	float x1 = numeric_limits<float>::quiet_NaN(), x2=x1, x3=x1, x4=x1, x5=x1, x6=x1, x7=x1, x8=x1, x9=x1;
	if (!PyArg_ParseTuple(args,"|fffffffff", &x1, &x2, &x3, &x4, &x5, &x6, &x7, &x8, &x9))
		return -1;
	
	if (!c_isnan(x1)) {
		self->data[0] = x1;
		if (!c_isnan(x2) && !c_isnan(x3) && !c_isnan(x4) && !c_isnan(x5) && !c_isnan(x6) && !c_isnan(x7) && !c_isnan(x8) && !c_isnan(x9)) {
			self->data[1] = x2;
			self->data[2] = x3;
			self->data[3] = x4;
			self->data[4] = x5;
			self->data[5] = x6;
			self->data[6] = x7;
			self->data[7] = x8;
			self->data[8] = x9;
			printf("x9 is %f\n", self->data[8]);

		} else {
			if (!c_isnan(x2) || !c_isnan(x3) || !c_isnan(x4) || !c_isnan(x5) || !c_isnan(x6) || !c_isnan(x7) || !c_isnan(x8) || !c_isnan(x9)) { errMsg("Invalid partial init of mat3"); }
			self->data[1] = self->data[2] = self->data[3] = self->data[4] = self->data[5] = self->data[6] = self->data[7] = self->data[8] = x1;
		}
	} else {
		/* Use identity init by default. */
		self->data[0] = 1.0; self->data[1] = 0.0; self->data[2] = 0.0;
		self->data[3] = 0.0; self->data[4] = 1.0; self->data[5] = 0.0;
		self->data[6] = 0.0; self->data[7] = 0.0; self->data[8] = 1.0;
	}
	return 0;
}

static PyObject* PbMat3Repr(PbMat3* self) {
	Manta::Matrix3x3f mat(self->data[0], self->data[1], self->data[2],
						  self->data[3], self->data[4], self->data[5],
						  self->data[6], self->data[7], self->data[8]);
	return PyUnicode_FromFormat(mat.toString().c_str());
}

PyTypeObject PbMat3Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"manta.mat3",             /* tp_name */
	sizeof(PbMat3),             /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)PbMat3Dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	(reprfunc)PbMat3Repr,      /* tp_repr */
	NULL, // &PbMat3NumberMethods,      /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
#if PY_MAJOR_VERSION >= 3
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE ,   /* tp_flags */
#else
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |  Py_TPFLAGS_CHECKTYPES,   /* tp_flags */
#endif
	"float matrix type",   /* tp_doc */
	0,                     /* tp_traverse */
	0,                     /* tp_clear */
	0,                     /* tp_richcompare */
	0,                     /* tp_weaklistoffset */
	0,                     /* tp_iter */
	0,                     /* tp_iternext */
	PbMat3Methods,             /* tp_methods */
	PbMat3Members,             /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PbMat3Init,      /* tp_init */
	0,                         /* tp_alloc */
	PbMat3New,                 /* tp_new */
};

// register

inline PyObject* castPy(PyTypeObject* p) {
	return reinterpret_cast<PyObject*>(static_cast<void*>(p));
}


// 2d matrix

extern PyTypeObject PbMat2Type;

struct PbMat2 {
	PyObject_HEAD
	float data[4];
};

static PyMethodDef PbMat2Methods[] = {
	{NULL}  // Sentinel
};

static PyMemberDef PbMat2Members[] = {
	{(char*)"v00", T_FLOAT, offsetof(PbMat2, data), 0, (char*)"v00"},
	{(char*)"v01", T_FLOAT, offsetof(PbMat2, data)+sizeof(float)*1, 0, (char*)"v01"},
	{(char*)"v10", T_FLOAT, offsetof(PbMat2, data)+sizeof(float)*2, 0, (char*)"v10"},
	{(char*)"v11", T_FLOAT, offsetof(PbMat2, data)+sizeof(float)*3, 0, (char*)"v11"},
	{NULL}  // Sentinel
};

static void PbMat2Dealloc(PbMat2* self) {
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * PbMat2New(PyTypeObject *type, PyObject *args, PyObject *kwds) {
	return type->tp_alloc(type, 0);
}

static int PbMat2Init(PbMat2 *self, PyObject *args, PyObject *kwds) {
	
	float x1 = numeric_limits<float>::quiet_NaN(), x2=x1, x3=x1, x4=x1;
	if (!PyArg_ParseTuple(args,"|ffff", &x1, &x2, &x3, &x4))
		return -1;
	
	if (!c_isnan(x1)) {
		self->data[0] = x1;
		if (!c_isnan(x2) && !c_isnan(x3) && !c_isnan(x4)) {
			self->data[1] = x2;
			self->data[2] = x3;
			self->data[3] = x4;
		} else {
			if (!c_isnan(x2) || !c_isnan(x3) || !c_isnan(x4)) { errMsg("Invalid partial init of mat2"); }
			self->data[1] = self->data[2] = self->data[3] = x1;
		}
	} else {
		/* Use identity init by default. */
		self->data[0] = 1.0; self->data[1] = 0.0;
		self->data[2] = 0.0; self->data[3] = 1.0;
	}
	return 0;
}

static PyObject* PbMat2Repr(PbMat2* self) {
	Manta::Matrix2x2f mat(self->data[0], self->data[1], self->data[2], self->data[3]);
	return PyUnicode_FromFormat(mat.toString().c_str());
}

PyTypeObject PbMat2Type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"manta.mat2",             /* tp_name */
	sizeof(PbMat2),             /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)PbMat2Dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	(reprfunc)PbMat2Repr,      /* tp_repr */
	NULL, // &PbMat2NumberMethods,      /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
#if PY_MAJOR_VERSION >= 3
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE ,   /* tp_flags */
#else
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |  Py_TPFLAGS_CHECKTYPES,   /* tp_flags */
#endif
	"float matrix type",   /* tp_doc */
	0,                     /* tp_traverse */
	0,                     /* tp_clear */
	0,                     /* tp_richcompare */
	0,                     /* tp_weaklistoffset */
	0,                     /* tp_iter */
	0,                     /* tp_iternext */
	PbMat2Methods,             /* tp_methods */
	PbMat2Members,             /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PbMat2Init,      /* tp_init */
	0,                         /* tp_alloc */
	PbMat2New,                 /* tp_new */
};

// register

void PbMatInitialize(PyObject* module) {
	if (PyType_Ready(&PbMat3Type) < 0) errMsg("can't initialize Matrix3x3 type");
	Py_INCREF(castPy(&PbMat3Type));
	PyModule_AddObject(module, "mat3", (PyObject *)&PbMat3Type);

	if (PyType_Ready(&PbMat2Type) < 0) errMsg("can't initialize Matrix2x2 type");
	Py_INCREF(castPy(&PbMat2Type));
	PyModule_AddObject(module, "mat2", (PyObject *)&PbMat2Type);
}
const static Pb::Register _REG(PbMatInitialize);

} // namespace
