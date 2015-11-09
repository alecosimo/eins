#if !defined(EINSVECIMPL_H)
#define EINSVECIMPL_H

#include <einsvec.h>
#include <petsc/private/vecimpl.h>

PETSC_INTERN PetscErrorCode VecMAXPY_Seq(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAYPX_Seq(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPY_Seq(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDot_Seq(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecMDot_Seq(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_Seq(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_Seq(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMTDot_Seq(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecNorm_Seq(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecSet_Seq(Vec,PetscScalar);

PETSC_EXTERN PetscErrorCode VecCreate_UNASM(Vec);

#endif/* EINSVECIMPL_H*/
