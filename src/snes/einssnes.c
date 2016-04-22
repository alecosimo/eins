#include <einsksp.h>
#include <private/einssnesimpl.h>
#include <petsc/private/snesimpl.h>


#undef __FUNCT__
#define __FUNCT__ "SNESNoJacobianIsComputed_default"
PETSC_EXTERN PetscErrorCode SNESNoJacobianIsComputed_default(SNES snes)
{
  PetscErrorCode ierr;
  KSP           ksp;
  
  PetscFunctionBegin;
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeJacobian(ksp,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SNESNoJacobianIsComputed"
/*@
   SNESNoJacobianIsComputed - Call back for informing that no jacobian
   is being computed. It is generally used for linear problems.

   Input: 
.  snes - the SNES context

   Level: basic

.keywords: SNES

.seealso: SNESCreate
@*/
PETSC_EXTERN PetscErrorCode SNESNoJacobianIsComputed(SNES snes)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscUseMethod(snes,"SNESNoJacobianIsComputed_C",(SNES),(snes));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  
  

