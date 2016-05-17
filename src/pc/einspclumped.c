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
  PetscErrorCode   ierr;
  Subdomain        sd;
  FETIMat_ctx      mat_ctx;
  PetscBool        flg;
  Mat              F;
  FETI             ft   = NULL;

  PetscFunctionBegin;
    /* get reference to the FETI context */
  F = pc->pmat;
  ierr = PetscObjectTypeCompare((PetscObject)F,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Cannot use preconditioner with non-shell matrix");
  ierr = MatShellUnAsmGetContext(F,(void**)&mat_ctx);CHKERRQ(ierr);
  /* there is no need to increment the reference to the FETI context because we are already referencing matrix F */
  ft   = mat_ctx->ft; 
  if (!ft) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix F is missing the FETI context");
  PetscValidHeaderSpecific(ft,FETI_CLASSID,0);

  pcl->ft   = ft;
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
  ierr = MatDestroy(&pcl->A_BB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCDestroy_LUMPED"
static PetscErrorCode PCDestroy_LUMPED(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",NULL);CHKERRQ(ierr);
  ierr = PCReset_LUMPED(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApplyLocal_LUMPED"
static PetscErrorCode PCApplyLocal_LUMPED(PC pc,Vec x,Vec y)
{
  PCFT_LUMPED      *pcl = (PCFT_LUMPED*)pc->data;
  PetscErrorCode   ierr;
  FETI             ft   = pcl->ft;
  Subdomain        sd   = ft->subdomain;
  Vec              lambda_local,y_local;
  
  PetscFunctionBegin;
  ierr = VecUnAsmGetLocalVectorRead(x,&lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(y,&y_local);CHKERRQ(ierr);
  /* Application of B_Ddelta^T */
  ierr = MatMultTranspose(ft->B_Ddelta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  ierr = MatMult(pcl->A_BB,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y_local);CHKERRQ(ierr);
  ierr = VecExchangeBegin(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(x,lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(y,y_local);CHKERRQ(ierr);
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
  Vec              lambda_local,y_local;
  
  PetscFunctionBegin;
  ierr = VecUnAsmGetLocalVectorRead(x,&lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(y,&y_local);CHKERRQ(ierr);
  /* Application of B_Ddelta^T */
  ierr = MatMultTranspose(ft->B_Ddelta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  ierr = MatMult(pcl->A_BB,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y_local);CHKERRQ(ierr);
  ierr = VecExchangeBegin(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(x,lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(y,y_local);CHKERRQ(ierr);
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
  pcl->work_vecs               = 0;
  pcl->s_reqs                  = 0;
  pcl->r_reqs                  = 0;
  pcl->isindex                 = 0;
  
  pc->ops->setup               = PCSetUp_LUMPED;
  pc->ops->reset               = PCReset_LUMPED;
  pc->ops->destroy             = PCDestroy_LUMPED;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->apply               = PCApply_LUMPED;
  pc->ops->applytranspose      = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",PCApplyLocal_LUMPED);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
EXTERN_C_END
