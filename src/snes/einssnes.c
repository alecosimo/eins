#include <petsc/private/snesimpl.h>


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
PETSC_EXTERN PetscErrorCode SNESNoJacobianIsComputed(SNES snes);
PETSC_EXTERN PetscErrorCode SNESNoJacobianIsComputed(SNES snes)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscUseMethod(snes,"SNESNoJacobianIsComputed_C",(SNES),(snes));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  
  

