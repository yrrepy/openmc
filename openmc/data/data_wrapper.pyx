# cython: language_level=3

cdef extern from "openmc/material.h" namespace "openmc":
    cdef cppclass Material:
        Material() except +
        void set_version(int)
        int version() const

cdef class PyMaterial:
    cdef Material* thisptr  # hold a C++ instance which we're wrapping

    def __cinit__(self):
        self.thisptr = new Material()

    def __dealloc__(self):
        del self.thisptr

    cpdef set_version(self, int version):
        self.thisptr.set_version(version)

    cpdef int get_version(self):
        return self.thisptr.version()

    property version:
        def __get__(self):
            return self.get_version()
        def __set__(self, int version):
            self.set_version(version)

    def __init__(self, int initial_version):
        self.version = initial_version 
