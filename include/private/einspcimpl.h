#if !defined(EINSPCIMPL_H)
#define EINSPCIMPL_H

#include <einspc.h>
#include <einsfeti.h>
#include <petsc/private/pcimpl.h>

#define PCFETIHEADER  \
  FETI ft; 

PETSC_EXTERN PetscErrorCode PCCreate_DIRICHLET(PC);

#endif/* EINSPCIMPL_H*/
