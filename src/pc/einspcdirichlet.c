#include <private/einspcimpl.h>

typedef struct {
  PCFETIHEADER
  Mat Sj;
} PCFT_DIRICHLET;


#undef  __FUNCT__
#define __FUNCT__ "PCSetUp_DIRICHLET"
static PetscErrorCode PCSetUp_DIRICHLET(PC pc)
{

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCDestroy_DIRICHLET"
static PetscErrorCode PCDestroy_DIRICHLET(PC pc)
{

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApply_DIRICHLET"
static PetscErrorCode PCApply_DIRICHLET(PC pc,Vec x,Vec y)
{

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "PCCreate_DIRICHLET"
/*@
   PCFETI_DIRCHLET - Implements the Dirchlet preconditioner for solving iteratively the FETI interface problem.

   Input Parameters:
.  pc - the PC context

@*/
PetscErrorCode PCCreate_DIRICHLET(PC pc)
{
  PCFT_DIRICHLET  *pcdr = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&pcdr);CHKERRQ(ierr);
  pc->data = (void*)pcdr;

  pc->ops->setup               = PCSetUp_DIRICHLET;
  pc->ops->reset               = NULL;
  pc->ops->destroy             = PCDestroy_DIRICHLET;
  pc->ops->setfromoptions      = NULL;
  pc->ops->view                = NULL;
  pc->ops->apply               = PCApply_DIRICHLET;
  pc->ops->applytranspose      = NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
