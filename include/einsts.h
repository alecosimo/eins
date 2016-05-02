#if !defined(EINSTS_H)
#define EINSTS_H
#include <petscts.h>

PETSC_EXTERN PetscErrorCode TSPostStepLinear(TS);
PETSC_EXTERN PetscErrorCode TSSetPostStepLinear(TS);
PETSC_EXTERN PetscErrorCode TSComputeI2JacobianConstantInvariant(TS,PetscReal,Vec,Vec,Vec,PetscReal,PetscReal,Mat,Mat,void*);

#endif /* EINSTS_H*/
