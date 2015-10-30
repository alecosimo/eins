#if !defined(EINSKSP_H)
#define EINSKSP_H

#include <petscksp.h>

#define KSPPJGMRES "pjgmres"

PETSC_EXTERN PetscErrorCode KSPSetProjection(KSP,PetscErrorCode (*)(void*,Vec,Vec),void*);
PETSC_EXTERN PetscErrorCode KSPSetReProjection(KSP,PetscErrorCode (*)(void*,Vec,Vec),void*);
PETSC_EXTERN PetscErrorCode KSPPJGMRESMonitorKrylov(KSP,PETSC_UNUSED PetscInt,PETSC_UNUSED PetscReal,void*);

#endif/* EINSKSP_H*/
