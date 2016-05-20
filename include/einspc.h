#if !defined(EINSPC_H)
#define EINSPC_H

#include <petscpc.h>

#define PCFETI_DIRICHLET "pcfeti_dirichlet"
#define PCFETI_LUMPED    "pcfeti_lumped"

PETSC_EXTERN PetscErrorCode PCApplyLocalWithPolling(PC,Vec,Vec,PetscInt*);
PETSC_EXTERN PetscErrorCode PCApplyLocal(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCPreApply(PC);
  
#endif/* EINSPC_H*/
