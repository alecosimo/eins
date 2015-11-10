#if !defined(EINSTEST_H)
#define EINSTEST_H

#include <petscvec.h>

PETSC_EXTERN PetscErrorCode TestAssertScalars(PetscScalar,PetscScalar,PetscScalar);
PETSC_EXTERN PetscErrorCode TestAssertVectors(Vec,Vec,PetscScalar);
PETSC_EXTERN PetscErrorCode TestAssertVectorLeTol(Vec,PetscScalar);
  
#endif/* EINSTEST_H*/
