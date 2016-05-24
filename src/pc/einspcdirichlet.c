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
  pcd->ft  = mat_ctx->ft;
  if (!pcd->ft) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix F is missing the FETI context");
  PetscValidHeaderSpecific(pcd->ft,FETI_CLASSID,0);
	
  sd = (pcd->ft)->subdomain;
  /* set KSP for solving the Dirchlet problem */
  if (!pcd->ksp_D) {
    ierr = SubdomainComputeSubmatrices(sd,MAT_INITIAL_MATRIX,PETSC_FALSE);CHKERRQ(ierr);
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

    ierr = KSPSetOperators(pcd->ksp_D,sd->A_II,sd->A_II);CHKERRQ(ierr);
    ierr = KSPSetUp(pcd->ksp_D);CHKERRQ(ierr);

    /* create local Schur complement matrix */
    ierr = MatCreateSchurComplement(sd->A_II,sd->A_II,sd->A_IB,sd->A_BI,sd->A_BB,&pcd->Sj);CHKERRQ(ierr);
    ierr = MatSchurComplementSetKSP(pcd->Sj,pcd->ksp_D);CHKERRQ(ierr);
  } else {
    ierr = SubdomainComputeSubmatrices(sd,MAT_REUSE_MATRIX,PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcd->ksp_D,sd->A_II,sd->A_II);CHKERRQ(ierr);
    ierr = KSPSetUp(pcd->ksp_D);CHKERRQ(ierr);
  }
  ierr = PCSetReusePreconditioner(pc,PETSC_FALSE);CHKERRQ(ierr);  
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
  if(pcd->work_vecs) { ierr = PCDeAllocateFETIWorkVecs_Private(pc);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCDestroy_DIRICHLET"
static PetscErrorCode PCDestroy_DIRICHLET(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocalWithPolling_C",NULL);CHKERRQ(ierr);

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
 PCFT_DIRICHLET      *pcd = (PCFT_DIRICHLET*)pc->data;
  PetscErrorCode      ierr;
  FETI                ft   = pcd->ft;
  Subdomain           sd   = ft->subdomain;
  PetscMPIInt         i_mpi;
  PetscInt            i;
  Vec                 vec,vec_res,vec_aux;
  const PetscScalar   *array_s;

  PetscFunctionBegin;
  /* allocate resources if not available */
  if(!pcd->work_vecs) { ierr = PCAllocateFETIWorkVecs_Private(pc,ft);CHKERRQ(ierr);}

  /* Application of B_Ddelta^T */
  ierr = MatMultTranspose(ft->B_Ddelta,x,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local Schur complement */
  ierr = MatMult(pcd->Sj,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y);CHKERRQ(ierr);

  /* send to my neighbors my local vector to which my neighbors' preconditioner must be applied */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = VecGetSubVector(x,pcd->isindex[i-1],&vec);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
    ierr = MPI_Isend(array_s,ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->s_reqs[i-1]);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = VecRestoreSubVector(x,pcd->isindex[i-1],&vec);CHKERRQ(ierr);
  }
  /* receive vectors from my neighbors */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Irecv(pcd->work_vecs[i-1],ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->r_reqs[i-1]);CHKERRQ(ierr);    
  }
  ierr = MPI_Waitall(pcd->n_reqs,pcd->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(pcd->n_reqs,pcd->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  /* apply preconditioner to received vectors */
  ierr = VecDuplicate(pcd->vec1,&vec_res);CHKERRQ(ierr);
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = VecSet(pcd->vec1,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(pcd->vec1,ft->n_shared_lb[i],ft->shared_lb[i],pcd->work_vecs[i-1],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(pcd->vec1);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcd->vec1);CHKERRQ(ierr);
    /* Application of B_Ddelta^T */
    ierr = MatMultTranspose(ft->B_Ddelta,pcd->vec1,sd->vec1_B);CHKERRQ(ierr);
    /* Application of local Schur complement */
    ierr = MatMult(pcd->Sj,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
    /* Application of B_Ddelta */
    ierr = MatMult(ft->B_Ddelta,sd->vec2_B,vec_res);CHKERRQ(ierr);
    /* communicate result */
    ierr = VecGetSubVector(vec_res,pcd->isindex[i-1],&vec_aux);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
    ierr = MPI_Isend(array_s,ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->s_reqs[i-1]);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
    ierr = VecRestoreSubVector(vec_res,pcd->isindex[i-1],&vec_aux);CHKERRQ(ierr);   
  } 
  /* receive results */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Irecv(pcd->work_vecs[i-1],ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->r_reqs[i-1]);CHKERRQ(ierr);    
  }
  ierr = MPI_Waitall(pcd->n_reqs,pcd->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(pcd->n_reqs,pcd->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_res);CHKERRQ(ierr);
  /* sum results of my neighbors */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = VecSet(pcd->vec1,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(pcd->vec1,ft->n_shared_lb[i],ft->shared_lb[i],pcd->work_vecs[i-1],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(pcd->vec1);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcd->vec1);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,pcd->vec1);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApplyLocalWithPolling_DIRICHLET"
static PetscErrorCode PCApplyLocalWithPolling_DIRICHLET(PC pc,Vec x,Vec y,PetscInt *_n2c)
{
  PCFT_DIRICHLET      *pcd = (PCFT_DIRICHLET*)pc->data;
  PetscErrorCode      ierr;
  FETI                ft   = pcd->ft;
  Subdomain           sd   = ft->subdomain;
  PetscMPIInt         i_mpi;
  PetscInt            i,is,ir;
  Vec                 vec,vec_res,vec_aux;
  PetscInt            n2c,n2c0;
  const PetscScalar   *array_s;
  
  PetscFunctionBegin;
  /* allocate resources if not available */
  if(!pcd->work_vecs) { ierr = PCAllocateFETIWorkVecs_Private(pc,ft);CHKERRQ(ierr);}

  n2c  = ((!x)==0);
  n2c0 = n2c;
  ierr = PCAllocateCommunication_Private(pc,&n2c);CHKERRQ(ierr);
  if (_n2c) *_n2c = n2c;

  if (n2c) {
    is = 0;
    if (n2c0) {
      /* Application of B_Ddelta^T */
      ierr = MatMultTranspose(ft->B_Ddelta,x,sd->vec1_B);CHKERRQ(ierr);
      /* Application of local Schur complement */
      ierr = MatMult(pcd->Sj,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
      /* Application of B_Ddelta */
      ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y);CHKERRQ(ierr);

      /* send to my neighbors my local vector to which my neighbors' preconditioner must be applied */
      for (is=0; is<ft->n_neigh_lb-1; is++){
	ierr = VecGetSubVector(x,pcd->isindex[is],&vec);CHKERRQ(ierr);
	ierr = VecGetArrayRead(vec,&array_s);CHKERRQ(ierr);   
	ierr = PetscMPIIntCast(ft->neigh_lb[is+1],&i_mpi);CHKERRQ(ierr);   
	ierr = MPI_Isend(array_s,ft->n_shared_lb[is+1],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->s_reqs[is]);CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(vec,&array_s);CHKERRQ(ierr);   
	ierr = VecRestoreSubVector(x,pcd->isindex[is],&vec);CHKERRQ(ierr);
      }
    }
    /* receive vectors from my neighbors */
    ir = 0;
    for (i=0; i<ft->n_neigh_lb-1; i++){
      if (pcd->pnc[i]) {
	ierr = PetscMPIIntCast(ft->neigh_lb[i+1],&i_mpi);CHKERRQ(ierr);
	ierr = MPI_Irecv(pcd->work_vecs[i],ft->n_shared_lb[i+1],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->r_reqs[ir++]);CHKERRQ(ierr);
      }
    }
    ierr = MPI_Waitall(ir,pcd->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    if(is) { ierr = MPI_Waitall(is,pcd->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}

    /* apply preconditioner to received vectors */
    ierr = VecDuplicate(pcd->vec1,&vec_res);CHKERRQ(ierr);
    is   = 0;
    for (i=0; i<ft->n_neigh_lb-1; i++){
      if (pcd->pnc[i]) {
	ierr = VecSet(pcd->vec1,0.0);CHKERRQ(ierr);
	ierr = VecSetValues(pcd->vec1,ft->n_shared_lb[i+1],ft->shared_lb[i+1],pcd->work_vecs[i],INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(pcd->vec1);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(pcd->vec1);CHKERRQ(ierr);
	/* Application of B_Ddelta^T */
	ierr = MatMultTranspose(ft->B_Ddelta,pcd->vec1,sd->vec1_B);CHKERRQ(ierr);
	/* Application of local Schur complement */
	ierr = MatMult(pcd->Sj,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
	/* Application of B_Ddelta */
	ierr = MatMult(ft->B_Ddelta,sd->vec2_B,vec_res);CHKERRQ(ierr);
	/* communicate result */
	ierr = VecGetSubVector(vec_res,pcd->isindex[i],&vec_aux);CHKERRQ(ierr);
	ierr = VecGetArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
	ierr = PetscMPIIntCast(ft->neigh_lb[i+1],&i_mpi);CHKERRQ(ierr);   
	ierr = MPI_Isend(array_s,ft->n_shared_lb[i+1],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->s_reqs[is++]);CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
	ierr = VecRestoreSubVector(vec_res,pcd->isindex[i],&vec_aux);CHKERRQ(ierr);
      }
    } 
    /* receive results */
    ir = 0;
    if (n2c0) {
      for (ir=0; ir<ft->n_neigh_lb-1; ir++){
	ierr = PetscMPIIntCast(ft->neigh_lb[ir+1],&i_mpi);CHKERRQ(ierr);
	ierr = MPI_Irecv(pcd->work_vecs[ir],ft->n_shared_lb[ir+1],MPIU_SCALAR,i_mpi,0,pcd->comm,&pcd->r_reqs[ir]);CHKERRQ(ierr);    
      }
    }
    if (ir) { ierr = MPI_Waitall(ir,pcd->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
    ierr = MPI_Waitall(is,pcd->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_res);CHKERRQ(ierr);

    if(n2c0) {
      /* sum results of my neighbors */
      for (i=0; i<ft->n_neigh_lb-1; i++){
	ierr = VecSet(pcd->vec1,0.0);CHKERRQ(ierr);
	ierr = VecSetValues(pcd->vec1,ft->n_shared_lb[i+1],ft->shared_lb[i+1],pcd->work_vecs[i],INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(pcd->vec1);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(pcd->vec1);CHKERRQ(ierr);
	ierr = VecAXPY(y,1.0,pcd->vec1);CHKERRQ(ierr);
      }
    }
  }
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
  pcd->work_vecs               = 0;
  pcd->s_reqs                  = 0;
  pcd->r_reqs                  = 0;
  pcd->isindex                 = 0;
    
  pc->ops->setup               = PCSetUp_DIRICHLET;
  pc->ops->reset               = PCReset_DIRICHLET;
  pc->ops->destroy             = PCDestroy_DIRICHLET;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->apply               = PCApply_DIRICHLET;
  pc->ops->applytranspose      = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocalWithPolling_C",PCApplyLocalWithPolling_DIRICHLET);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",PCApplyLocal_DIRICHLET);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
EXTERN_C_END
