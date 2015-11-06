#include <private/einspcimpl.h>
#include <private/einsfetiimpl.h>
#include <einssys.h>


typedef struct {
  PCFETIHEADER
  Mat         A_BB; /* reference to A_BB: so it must be destroyed */
} PCFT_LUMPED;


#undef  __FUNCT__
#define __FUNCT__ "PCSetUp_LUMPED"
static PetscErrorCode PCSetUp_LUMPED(PC pc)
{
  PCFT_LUMPED      *pcl = (PCFT_LUMPED*)pc->data;
  FETI             ft   = NULL;
  PetscErrorCode   ierr;
  Mat              F;
  Subdomain        sd;
  
  PetscFunctionBegin;
  /* get reference to the FETI context */
  F = pc->pmat;
  ierr = PetscObjectQuery((PetscObject)F,"FETI",(PetscObject*)&ft);CHKERRQ(ierr);
  if (!ft) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix F is missing the FETI context");
  PetscValidHeaderSpecific(ft,FETI_CLASSID,0);
  pcl->ft   = ft;
  ierr      = PetscObjectReference((PetscObject)ft);CHKERRQ(ierr);
  sd        = ft->subdomain;
  ierr      = SubdomainComputeSubmatrices(sd,MAT_INITIAL_MATRIX,PETSC_TRUE);CHKERRQ(ierr);
  pcl->A_BB = sd->A_BB;
  ierr      = PetscObjectReference((PetscObject)sd->A_BB);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCReset_LUMPED"
static PetscErrorCode PCReset_LUMPED(PC pc)
{
  PCFT_LUMPED    *pcl = (PCFT_LUMPED*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = FETIDestroy(&pcl->ft);CHKERRQ(ierr);
  ierr = MatDestroy(&pcl->A_BB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCDestroy_LUMPED"
static PetscErrorCode PCDestroy_LUMPED(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCReset_LUMPED(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApply_LUMPED"
static PetscErrorCode PCApply_LUMPED(PC pc,Vec x,Vec y)
{
  PCFT_LUMPED      *pcl = (PCFT_LUMPED*)pc->data;
  PetscErrorCode   ierr;
  FETI             ft   = pcl->ft;
  Subdomain        sd   = ft->subdomain;

  PetscFunctionBegin;
  /* Application of B_Ddelta^T */
  ierr = VecScatterBegin(ft->l2g_lambda,x,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,x,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSet(sd->vec1_B,0.0);CHKERRQ(ierr);
  ierr = MatMultTranspose(ft->B_Ddelta,ft->lambda_local,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  ierr = MatMult(pcl->A_BB,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,ft->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ft->l2g_lambda,ft->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,ft->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "PCCreate_LUMPED"
/*@
   PCFETI_LUMPED - Implements the Lumped preconditioner for solving iteratively the FETI interface problem.

   Input Parameters:
.  pc - the PC context

@*/
PetscErrorCode PCCreate_LUMPED(PC pc)
{
  PCFT_LUMPED     *pcl = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&pcl);CHKERRQ(ierr);
  pc->data = (void*)pcl;

  pcl->ft                      = 0;
  pcl->A_BB                    = 0;
  
  pc->ops->setup               = PCSetUp_LUMPED;
  pc->ops->reset               = PCReset_LUMPED;
  pc->ops->destroy             = PCDestroy_LUMPED;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->apply               = PCApply_LUMPED;
  pc->ops->applytranspose      = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
