#if !defined(EINSKSP_H)
#define EINSKSP_H

#include <petscksp.h>

#define KSPPJGMRES "pjgmres"

PETSC_EXTERN PetscErrorCode KSPSetProjection(KSP,PetscErrorCode (*)(void*,Vec,Vec),void*);
PETSC_EXTERN PetscErrorCode KSPSetReProjection(KSP,PetscErrorCode (*)(void*,Vec,Vec),void*);

#endif/* EINSKSP_H*/
