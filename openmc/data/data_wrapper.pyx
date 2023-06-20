# distutils: language = c++
# distutils: sources = Material.cpp

cdef extern from "openmc/Material.h" namespace "openmc":
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

    cdef set_version(self, int version):
        self.thisptr.set_version(version)

    cdef int version(self):
        return self.thisptr.version()

    property version:
        def __get__(self):
            return self.version()
        def __set__(self, int version):
            self.set_version(version)