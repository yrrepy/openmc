cdef extern from "openmc/include/material.h":
    cdef int version()
    cdef void set_version(int version)

cdef class DataWrapper:
    cdef void setVersion(self, int version):
        set_version(version)

    cdef int getVersion(self):
        return version()

    property version:
        def __get__(self):
            return self.getVersion()
        def __set__(self, int version):
            self.setVersion(version)

    def __cinit__(self, int initial_version):
        self.setVersion(initial_version)
