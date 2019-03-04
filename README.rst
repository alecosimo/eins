EINS: Efficient Integrators and Non-linear Solvers
==================================================

Overview
--------

EINS is a library for experimenting with FETI (Finite Element Tearing
and Interconnect) solvers. These solvers could be involved in the
numerical simulation of static and dynamic problems, which at the same
time could be linear or non-linear. I decided based the implement as
extensions of the PETSc library (version 3.7). Some comments on the
design of the code can be found [here](https://cimec.org.ar/ojs/index.php/mc/article/view/5046). 

This is an experimental code which I have some time without working on
it. Therefore, if you find it interesting for your personal usage and
have some question or want to discuss about the philosophy behind the design
of the code, do not hesitate to contact me. I have some ideas that I
didn't implement and would help to improve the quality of the
code.

CMake is used to control the software compilation and testing. In
order to be able to successfully compile the library, you additionally
need to have installed the SLEPc library (version 3.6.3). Once you
compiled the library, you can issue -make check- in order to run the
tests. 
