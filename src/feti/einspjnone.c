#include <private/einsfetiimpl.h>

#undef __FUNCT__
#define __FUNCT__ "FETIPJComputeInitialCondition_PJNONE"
static PetscErrorCode FETIPJComputeInitialCondition_PJNONE(PETSC_UNUSED FETIPJ ftpj,PETSC_UNUSED Vec x,Vec y)
{
  PetscErrorCode    ierr; 
  PetscFunctionBegin;
  ierr = VecSet(y, 0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJCreate_PJNONE"
PETSC_EXTERN PetscErrorCode FETIPJCreate_PJNONE(FETIPJ);
PetscErrorCode FETIPJCreate_PJNONE(FETIPJ ftpj)
{
  PetscFunctionBegin;
  ftpj->data = 0;
  ftpj->ops->setup               = 0;
  ftpj->ops->destroy             = 0;
  ftpj->ops->setfromoptions      = 0;
  ftpj->ops->gatherneighbors     = 0;
  ftpj->ops->assemble            = 0;
  ftpj->ops->factorize           = 0;
  ftpj->ops->initialcondition    = FETIPJComputeInitialCondition_PJNONE;
  ftpj->ops->computealpha        = 0;
  PetscFunctionReturn(0);  
}


