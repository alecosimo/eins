#include <einssnes.h>
#include <einsts.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#else
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#endif

#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian2ConstantInvariant"
/*@
   TSComputeIJacobian2ConstantInvariant - Reuses the Jacobian already
   computed. That is, it does not compute anything and it does not
   modify the already computed Jacobian.

@*/
PETSC_EXTERN PetscErrorCode TSComputeIJacobian2ConstantInvariant(PETSC_UNUSED TS ts,PETSC_UNUSED PetscReal b,
								 PETSC_UNUSED Vec c, PETSC_UNUSED Vec d, PETSC_UNUSED Vec e,
								 PETSC_UNUSED PetscReal f, PETSC_UNUSED PetscReal g, PETSC_UNUSED Mat h,
								 PETSC_UNUSED Mat i, PETSC_UNUSED void* j)
{
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  ierr = SNESNoJacobianIsComputed(ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
