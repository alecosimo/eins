#include <einssnes.h>
#include <einsts.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#else
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#endif

#undef __FUNCT__
#define __FUNCT__ "TSComputeI2JacobianConstantInvariant"
/*@
   TSComputeIJacobian2ConstantInvariant - Reuses the Jacobian already
   computed. That is, it does not compute anything and it does not
   modify the already computed Jacobian.

@*/
PETSC_EXTERN PetscErrorCode TSComputeI2JacobianConstantInvariant(PETSC_UNUSED TS ts,PETSC_UNUSED PetscReal b,
								 PETSC_UNUSED Vec c, PETSC_UNUSED Vec d, PETSC_UNUSED Vec e,
								 PETSC_UNUSED PetscReal f, PETSC_UNUSED PetscReal g, PETSC_UNUSED Mat h,
								 PETSC_UNUSED Mat i, PETSC_UNUSED void* j)
{
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  ierr = SNESSetComputeJacobian(ts->snes,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSPostStepLinear"
/*@
   TSPostStepLinear - PostStep function for linear problems.

.seealso:  TS, TSCreate(), TSSetPostStepLinear()
@*/
PETSC_EXTERN PetscErrorCode TSPostStepLinear(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  /* preventing linear problems from recomputing the jacobian */
  if(ts->problem_type == TS_LINEAR) {
    Mat   J,P;
    void* ctx;
    ierr = TSGetIJacobian(ts,&J,&P,NULL,&ctx);CHKERRQ(ierr);
    ierr = TSSetI2Jacobian(ts,J,P,TSComputeI2JacobianConstantInvariant,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetPostStepLinear"
/*@
   TSSetPostStepLinear - Sets up TS for a linear problem. Mainly it sets the
   PostStep and PreStep functions. You can instead in your own
   PostStep function call at the begginging to
   TSPostStepLinear()

.seealso:  TS, TSCreate(), TSPostStepLinear()
@*/
PETSC_EXTERN PetscErrorCode TSSetPostStepLinear(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  ierr = TSSetPostStep(ts,TSPostStepLinear);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
