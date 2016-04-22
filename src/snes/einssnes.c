#include <einsksp.h>
#include <private/einssnesimpl.h>
#include <petsc/private/snesimpl.h>


#undef __FUNCT__
#define __FUNCT__ "SNESSetComputeJacobian_default"
PETSC_EXTERN PetscErrorCode SNESSetComputeJacobian_default(SNES snes,PetscBool flg)
{
  PetscErrorCode ierr;
  KSP           ksp;
  
  PetscFunctionBegin;
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeJacobian(ksp,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SNESSetComputeJacobian"
/*@
   SNESSetComputeJacobian - Call back for setting if the jacobian
   should not be computed. It is generally used for linear problems.

   Input: 
+  snes - the SNES context
-  flg  - Boolean value

   Level: basic

.keywords: SNES

.seealso: SNESCreate
@*/
PETSC_EXTERN PetscErrorCode SNESSetComputeJacobian(SNES snes,PetscBool flg)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscUseMethod(snes,"SNESSetComputeJacobian_C",(SNES,PetscBool),(snes,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  
  

