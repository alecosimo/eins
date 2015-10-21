#if !defined(EINSPCIMPL_H)
#define EINSPCIMPL_H

#include <einspc.h>
#include <petsc/private/pcimpl.h>

#define PCFETIHEADER                                                  \
  Mat mat; 

PETSC_EXTERN PetscErrorCode PCCreate_DIRICHLET(PC);

#endif/* EINSPCIMPL_H*/
