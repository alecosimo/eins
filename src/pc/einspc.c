#include <private/einspcimpl.h>


#undef __FUNCT__
#define __FUNCT__ "KSPSetComputeJacobian"
/*@
   KSPSetComputeJacobian - Sets the variable controlling the computation of the jacobian.

   Input Parameters:
.  ksp - the KSP context
.  flg - flag to set

   Level: intermediate

@*/
PETSC_EXTERN PetscErrorCode PCApplyLocal(PC pc,Vec x, Vec y) {
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ierr = PetscUseMethod(pc,"PCApplyLocal_C",(PC,Vec,Vec),(pc,x,y));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
