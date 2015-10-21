#include <private/einspcimpl.h>
#include <private/einsfetiimpl.h>
#include <einssys.h>


typedef struct {
  PCFETIHEADER
  Mat Sj;
  KSP ksp_D;
} PCFT_DIRICHLET;


#undef  __FUNCT__
#define __FUNCT__ "PCSetUp_DIRICHLET"
static PetscErrorCode PCSetUp_DIRICHLET(PC pc)
{
  PCFT_DIRICHLET  *pcd = (PCFT_DIRICHLET*)pc->data;
  FETI             ft  = NULL;
  PetscErrorCode   ierr;
  Mat              F;
  Subdomain        sd;
  PetscBool        issbaij;
  PC               pctemp;
  
  PetscFunctionBegin;
  /* get reference to the FETI context */
  F = pc->pmat;
  ierr = PetscObjectQuery((PetscObject)F,"FETI",(PetscObject*)&ft);CHKERRQ(ierr);
  if (!ft) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix F is missing the FETI context");
  PetscValidHeaderSpecific(ft,FETI_CLASSID,0);
  pcd->ft = ft;
  ierr    = PetscObjectReference((PetscObject)ft);CHKERRQ(ierr);
  sd      = ft->subdomain;
  ierr    = SubdomainComputeSubmatrices(sd,MAT_INITIAL_MATRIX,PETSC_FALSE);CHKERRQ(ierr);

  /* set KSP for solving the Dirchlet problem */
  if (!pcd->ksp_D) {
    ierr = KSPCreate(PETSC_COMM_SELF,&pcd->ksp_D);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcd->ksp_D,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetType(pcd->ksp_D,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcd->ksp_D,"feti_pc_dirichlet_");CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(sd->A_II,"feti_pc_dirichlet_");CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)sd->A_II,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
    ierr = KSPGetPC(pcd->ksp_D,&pctemp);CHKERRQ(ierr);
    if (issbaij) {
      ierr = PCSetType(pctemp,PCCHOLESKY);CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pctemp,PCLU);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(pcd->ksp_D);CHKERRQ(ierr);
    ierr = PCFactorSetReuseFill(pctemp,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(pcd->ksp_D,sd->A_II,sd->A_II);CHKERRQ(ierr);
  ierr = KSPSetUp(pcd->ksp_D);CHKERRQ(ierr);
  
  /* create local Schur complement matrix */
  ierr = MatCreateSchurComplement(sd->A_II,sd->A_II,sd->A_IB,sd->A_BI,sd->A_BB,&pcd->Sj);CHKERRQ(ierr);
  ierr = MatSchurComplementSetKSP(pcd->Sj,pcd->ksp_D);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCReset_DIRICHLET"
static PetscErrorCode PCReset_DIRICHLET(PC pc)
{
  PCFT_DIRICHLET *pcd = (PCFT_DIRICHLET*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = FETIDestroy(&pcd->ft);CHKERRQ(ierr);
  ierr = MatDestroy(&pcd->Sj);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcd->ksp_D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCDestroy_DIRICHLET"
static PetscErrorCode PCDestroy_DIRICHLET(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCReset_DIRICHLET(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApply_DIRICHLET"
static PetscErrorCode PCApply_DIRICHLET(PC pc,Vec x,Vec y)
{
  PCFT_DIRICHLET   *pcd = (PCFT_DIRICHLET*)pc->data;
  PetscErrorCode   ierr;
  FETI             ft   = pcd->ft;
  Subdomain        sd   = ft->subdomain;

  PetscFunctionBegin;
  /* Application of B_Ddelta^T */
  ierr = VecScatterBegin(ft->l2g_lambda,x,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,x,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSet(sd->vec1_B,0.0);CHKERRQ(ierr);
  ierr = MatMultTranspose(ft->B_Ddelta,ft->lambda_local,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  ierr = MatMult(pcd->Sj,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,ft->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ft->l2g_lambda,ft->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,ft->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "PCCreate_DIRICHLET"
/*@
   PCFETI_DIRCHLET - Implements the Dirchlet preconditioner for solving iteratively the FETI interface problem.

   Input Parameters:
.  pc - the PC context

   Options database:
.  -feti_pc_dirichilet_<ksp or pc option>: options for the KSP or PC to use for solving the Dirichlet problem
   associated to the Dirichlet preconditioner

@*/
PetscErrorCode PCCreate_DIRICHLET(PC pc)
{
  PCFT_DIRICHLET  *pcd = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&pcd);CHKERRQ(ierr);
  pc->data = (void*)pcd;

  pcd->ksp_D                   = 0;
  pcd->ft                      = 0;
  pcd->Sj                      = 0;
  
  pc->ops->setup               = PCSetUp_DIRICHLET;
  pc->ops->reset               = PCReset_DIRICHLET;
  pc->ops->destroy             = PCDestroy_DIRICHLET;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->apply               = PCApply_DIRICHLET;
  pc->ops->applytranspose      = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
