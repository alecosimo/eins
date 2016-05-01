#include <einssnes.h>
#include <einsts.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#else
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include "tsalpha2impl.h"
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
PETSC_EXTERN PetscErrorCode TSPostStepLinear(PETSC_UNUSED TS ts)
{
  PETSC_UNUSED PetscErrorCode ierr;

  PetscFunctionBegin;
#if PETSC_VERSION_GE(3,7,0)
  /* preventing linear problems from recomputing the jacobian */
  if(ts->problem_type == TS_LINEAR) {
    Mat   J,P;
    void* ctx;
    ierr = TSGetIJacobian(ts,&J,&P,NULL,&ctx);CHKERRQ(ierr);
    ierr = TSSetI2Jacobian(ts,J,P,TSComputeI2JacobianConstantInvariant,ctx);CHKERRQ(ierr);
  }
#endif
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


#undef __FUNCT__
#define __FUNCT__ "TS3SetSolution"
/*@
   TS3SetSolution - Sets the initial solution and the first and second
   time-derivative vectors for use by the TSALPHA2 routines.

   Logically Collective on TS and Vec

   Input Parameters:
+  ts - the TS context
.  X - the solution vector
.  V - the first time-derivative vector
-  A - the second time-derivative vector

   Level: beginner

.keywords: TS, TSALPHA2, timestep, set, solution, initial conditions
@*/
PetscErrorCode TS3SetSolution(TS ts,Vec X,Vec V,Vec A)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(V,VEC_CLASSID,3);
  PetscValidHeaderSpecific(A,VEC_CLASSID,4);

  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)V);CHKERRQ(ierr);
  ierr = VecDestroy(&ts->vec_dot);CHKERRQ(ierr);
  ts->vec_dot = V;
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = VecDestroy(&th->A1);CHKERRQ(ierr);
  th->A1 = A;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TS3GetSolution"
/*@
   TSGetSolution3 - Returns the solution and first and second
   time-derivative vectors at the present timestep. It is valid to
   call this routine inside the function that you are evaluating in
   order to move to the new timestep. This vector not changed until
   the solution at the next timestep has been calculated.

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context

   Output Parameter:
+  X - the vector containing the solution
.  V - the vector containing the first time-derivative
-  A - the vector containing the second time-derivative

   Level: intermediate

.seealso: TSGetTimeStep()

.keywords: TS, TSALPHA2, timestep, get, solution
@*/
PetscErrorCode TS3GetSolution(TS ts,Vec *X, Vec *V, Vec *A)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (X) PetscValidPointer(X,2);
  if (V) PetscValidPointer(V,3);
  if (A) PetscValidPointer(A,4);

  if (!th->vec_dot && ts->vec_sol) {
    ierr = VecDuplicate(ts->vec_sol,&th->vec_dot);CHKERRQ(ierr);
  }
  ierr = TS2GetSolution(ts,X,V);CHKERRQ(ierr);
  if (A) {*A = th->A1;}
  PetscFunctionReturn(0);
}
