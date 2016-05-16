#include <private/einspcimpl.h>
#include <einspcdirichlet.h>
#include <private/einsfetiimpl.h>
#include <einssys.h>


#undef  __FUNCT__
#define __FUNCT__ "PCSetUp_DIRICHLET"
static PetscErrorCode PCSetUp_DIRICHLET(PC pc)
{
  PCFT_DIRICHLET  *pcd = (PCFT_DIRICHLET*)pc->data;
  PetscErrorCode   ierr;
  Subdomain        sd;
  PetscBool        issbaij;
  PC               pctemp;
  FETI             ft  = NULL;
  Mat              F;
  FETIMat_ctx      mat_ctx;
  PetscBool        flg;

  
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
  
  pcd->ft = ft;
  sd      = ft->subdomain;
  ierr    = SubdomainComputeSubmatrices(sd,MAT_INITIAL_MATRIX,PETSC_FALSE);CHKERRQ(ierr);

  /* set KSP for solving the Dirchlet problem */
  if (!pcd->ksp_D) {
    ierr = KSPCreate(PETSC_COMM_SELF,&pcd->ksp_D);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcd->ksp_D,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)pcd->ksp_D);CHKERRQ(ierr);
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

  //ac
  {
    PetscInt    rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);   
    PetscViewer viewer;
    if(!rank) {
      PetscViewerASCIIOpen(PETSC_COMM_SELF, "A_II.m", &viewer);
      PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
      MatView(sd->A_II,viewer);
      PetscViewerPopFormat(viewer);
      PetscViewerDestroy(&viewer);

      PetscViewerASCIIOpen(PETSC_COMM_SELF, "A_IB.m", &viewer);
      PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
      MatView(sd->A_IB,viewer);
      PetscViewerPopFormat(viewer);
      PetscViewerDestroy(&viewer);

      PetscViewerASCIIOpen(PETSC_COMM_SELF, "A_BI.m", &viewer);
      PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
      MatView(sd->A_BI,viewer);
      PetscViewerPopFormat(viewer);
      PetscViewerDestroy(&viewer);

      PetscViewerASCIIOpen(PETSC_COMM_SELF, "A_BB.m", &viewer);
      PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
      MatView(sd->A_BB,viewer);
      PetscViewerPopFormat(viewer);
      PetscViewerDestroy(&viewer);

    }
  }

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
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",NULL);CHKERRQ(ierr);
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
  Vec              lambda_local,y_local;
  
  PetscFunctionBegin;
  ierr = VecUnAsmGetLocalVectorRead(x,&lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(y,&y_local);CHKERRQ(ierr);
  /* Application of B_Ddelta^T */
  ierr = MatMultTranspose(ft->B_Ddelta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  ierr = MatMult(pcd->Sj,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y_local);CHKERRQ(ierr);
  ierr = VecExchangeBegin(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(x,lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(y,y_local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApplyLocal_DIRICHLET"
static PetscErrorCode PCApplyLocal_DIRICHLET(PC pc,Vec x,Vec y)
{
  PCFT_DIRICHLET   *pcd = (PCFT_DIRICHLET*)pc->data;
  PetscErrorCode   ierr;
  FETI             ft   = pcd->ft;
  Subdomain        sd   = ft->subdomain;
  Vec              lambda_local,y_local;
  MPI_Comm         comm;
  
  PetscFunctionBegin;
  ierr = VecUnAsmGetLocalVectorRead(x,&lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(y,&y_local);CHKERRQ(ierr);
  /* Application of B_Ddelta^T */
  ierr = MatMultTranspose(ft->B_Ddelta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  ierr = MatMult(pcd->Sj,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y_local);CHKERRQ(ierr);
  /* communication */
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  for (i=0; i<ft->n_neigh_lb; i++){
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
    ierr = VecGetSubVector(xi->vlocal,isindex,&vec);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
    ierr = MPI_Isend(array_s,ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&pcd->s_reqs[i]);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = VecRestoreSubVector(xi->vlocal,isindex,&vec);CHKERRQ(ierr);
    ierr = ISDestroy(&isindex);CHKERRQ(ierr);
  }

  for (i=0; i<ve->n_neigh; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Irecv(pcd->work_vecs[i],ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&pcd->r_reqs[i]);CHKERRQ(ierr);    
  }


  ierr = VecUnAsmRestoreLocalVectorRead(x,lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(y,y_local);CHKERRQ(ierr);
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

  pcd->ft                      = 0;
  pcd->ksp_D                   = 0;
  pcd->Sj                      = 0;
  
  pc->ops->setup               = PCSetUp_DIRICHLET;
  pc->ops->reset               = PCReset_DIRICHLET;
  pc->ops->destroy             = PCDestroy_DIRICHLET;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->apply               = PCApply_DIRICHLET;
  pc->ops->applytranspose      = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",PCApplyLocal_DIRICHLET);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
EXTERN_C_END
