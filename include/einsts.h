/* TSALPHA2 code originally obtained from petiga (petscts2.h): https://bitbucket.org/dalcinl/petiga */
#if !defined(EINSTS_H)
#define EINSTS_H
#include <petscts.h>

typedef PetscErrorCode (*TSIFunction2)(TS,PetscReal,Vec,Vec,Vec,Vec,void*);
typedef PetscErrorCode (*TSIJacobian2)(TS,PetscReal,Vec,Vec,Vec,PetscReal,PetscReal,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode TSSetIFunction2(TS,Vec,TSIFunction2,void*);
PETSC_EXTERN PetscErrorCode TSSetIJacobian2(TS,Mat,Mat,TSIJacobian2,void*);
PETSC_EXTERN PetscErrorCode TSComputeIFunction2(TS,PetscReal,Vec,Vec,Vec,Vec,PetscBool);
PETSC_EXTERN PetscErrorCode TSComputeIJacobian2(TS,PetscReal,Vec,Vec,Vec,PetscReal,PetscReal,Mat,Mat,PetscBool);
PETSC_EXTERN PetscErrorCode TSComputeIJacobian2ConstantInvariant(TS,PetscReal,Vec,Vec,Vec,PetscReal,PetscReal,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode TSSetSolution2(TS,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSGetSolution2(TS,Vec*,Vec*);
PETSC_EXTERN PetscErrorCode TSSetSolution3(TS,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSGetSolution3(TS,Vec*,Vec*,Vec*);
PETSC_EXTERN PetscErrorCode TSSolve2(TS,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSSolve3(TS,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSInterpolate2(TS,PetscReal,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSEvaluateStep2(TS,PetscInt,Vec,Vec,PetscBool*);

#define TSALPHA2 "alpha2"
PETSC_EXTERN PetscErrorCode TSAlpha2UseAdapt(TS,PetscBool);
PETSC_EXTERN PetscErrorCode TSAlpha2SetRadius(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2SetParams(TS,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2GetParams(TS,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

#endif /* EINSTS_H*/