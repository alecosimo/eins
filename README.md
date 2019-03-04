EINS: Efficient Integrators and Non-linear Solvers
==================================================

EINS is a library for experimenting with FETI-based solvers. The FETI
(Finite Element Tearing and Interconnect) method is a well-established
domain decomposition technique for solving large systems of equations
which usually result from the discretisation of partial differential
equations. Among the various versions of the FETI method, the
one-level FETI (FETI-1), the second-level FETI (FETI-2) and the
Dual-Primal FETI (FETI-DP) are the main versions from which further
improvements have been proposed. This library only considers the
family of the FETI-1 and FETI-2 methodologies which are characterised
by enforcing the continuity of the solution between subdomains using
Lagrange multipliers. Some comments on the design of the library can be
found [here](https://cimec.org.ar/ojs/index.php/mc/article/view/5046)

The different FETI methods are implemented as extensions to the [PETSc
library,](https://www.mcs.anl.gov/petsc/) therefore inheriting many of
the powerful features that this library for scientific computing
has. This is an experimental code which I have some time without
working on it. Therefore, if you find it interesting and have some
question or want to discuss about the philosophy behind the design of
the code, do not hesitate to contact me. I have some ideas that would
help to improve the quality of the code, but which I didn't implement.

CMake is used to control the software compilation and testing. In
order to be able to successfully compile the library, in addition to
the PETSc library (version 3.7), you need to have installed the [SLEPc
library](http://slepc.upv.es/) (version 3.6.3). Once you compile the
library, the tests can be run with the command *make check*. Just be
aware that you will not find so many examples. Tests dealing with
dynamic problems are not provided. If you are interested in this kind
of problems, I can give you a hint on how to use the library for that
purpose. 
