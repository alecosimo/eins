#if !defined(EINSSNESIMPL_H)
#define EINSSNESIMPL_H

#include <einssnes.h>

PETSC_EXTERN PetscErrorCode SNESCreate_FETIONLY(SNES);
PETSC_EXTERN PetscErrorCode SNESSetComputeJacobian_default(SNES,PetscBool);

#endif/* EINSSNESIMPL_H*/
