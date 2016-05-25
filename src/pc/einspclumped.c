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

  PetscFunctionBegin;
  /* get reference to the FETI context */
  F = pc->pmat;
  ierr = PetscObjectTypeCompare((PetscObject)F,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Cannot use preconditioner with non-shell matrix");
  ierr = MatShellUnAsmGetContext(F,(void**)&mat_ctx);CHKERRQ(ierr);
  /* there is no need to increment the reference to the FETI context because we are already referencing matrix F */
  pcl->ft  = mat_ctx->ft; 
  if (!pcl->ft) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix F is missing the FETI context");
  PetscValidHeaderSpecific(pcl->ft,FETI_CLASSID,0);

  sd = (pcl->ft)->subdomain;
  if (!pc->setupcalled) {
    ierr      = SubdomainComputeSubmatrices(sd,MAT_INITIAL_MATRIX,PETSC_TRUE);CHKERRQ(ierr);
    pcl->A_BB = sd->A_BB;
    ierr      = PetscObjectReference((PetscObject)sd->A_BB);CHKERRQ(ierr);
  } else {
    ierr      = SubdomainComputeSubmatrices(sd,MAT_REUSE_MATRIX,PETSC_TRUE);CHKERRQ(ierr);
    ierr      = MatDestroy(&pcl->A_BB);CHKERRQ(ierr);
    pcl->A_BB = sd->A_BB;
    ierr      = PetscObjectReference((PetscObject)sd->A_BB);CHKERRQ(ierr);
  }
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
  if(pcl->work_vecs) { ierr = PCDeAllocateFETIWorkVecs_Private(pc);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCDestroy_LUMPED"
static PetscErrorCode PCDestroy_LUMPED(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocalWithPolling_C",NULL);CHKERRQ(ierr);
  ierr = PCReset_LUMPED(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApplyLocal_LUMPED"
static PetscErrorCode PCApplyLocal_LUMPED(PC pc,Vec x,Vec y)
{
  PCFT_LUMPED         *pcl = (PCFT_LUMPED*)pc->data;
  PetscErrorCode      ierr;
  FETI                ft   = pcl->ft;
  Subdomain           sd   = ft->subdomain;
  PetscMPIInt         i_mpi;
  PetscInt            i;
  Vec                 vec,vec_res,vec_aux;
  const PetscScalar   *array_s;

  
  PetscFunctionBegin;
  /* allocate resources if not available */
  if(!pcl->work_vecs) { ierr = PCAllocateFETIWorkVecs_Private(pc,ft);CHKERRQ(ierr);}

  /* Application of B_Ddelta^T */
  ierr = MatMultTranspose(ft->B_Ddelta,x,sd->vec1_B);CHKERRQ(ierr);
  /* Application of local A_BB */
  ierr = MatMult(pcl->A_BB,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y);CHKERRQ(ierr);

  /* send to my neighbors my local vector to which my neighbors' preconditioner must be applied */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = VecGetSubVector(x,pcl->isindex[i-1],&vec);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
    ierr = MPI_Isend(array_s,ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->s_reqs[i-1]);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = VecRestoreSubVector(x,pcl->isindex[i-1],&vec);CHKERRQ(ierr);
  }
  /* receive vectors from my neighbors */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Irecv(pcl->work_vecs[i-1],ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->r_reqs[i-1]);CHKERRQ(ierr);    
  }
  ierr = MPI_Waitall(pcl->n_reqs,pcl->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(pcl->n_reqs,pcl->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  /* apply preconditioner to received vectors */
  ierr = VecDuplicate(pcl->vec1,&vec_res);CHKERRQ(ierr);
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = VecSet(pcl->vec1,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(pcl->vec1,ft->n_shared_lb[i],ft->shared_lb[i],pcl->work_vecs[i-1],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(pcl->vec1);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcl->vec1);CHKERRQ(ierr);
    /* Application of B_Ddelta^T */
    ierr = MatMultTranspose(ft->B_Ddelta,pcl->vec1,sd->vec1_B);CHKERRQ(ierr);
    /* Application of local A_BB */
    ierr = MatMult(pcl->A_BB,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
    /* Application of B_Ddelta */
    ierr = MatMult(ft->B_Ddelta,sd->vec2_B,vec_res);CHKERRQ(ierr);
    /* communicate result */
    ierr = VecGetSubVector(vec_res,pcl->isindex[i-1],&vec_aux);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
    ierr = MPI_Isend(array_s,ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->s_reqs[i-1]);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
    ierr = VecRestoreSubVector(vec_res,pcl->isindex[i-1],&vec_aux);CHKERRQ(ierr);   
  } 
  /* receive results */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Irecv(pcl->work_vecs[i-1],ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->r_reqs[i-1]);CHKERRQ(ierr);    
  }
  ierr = MPI_Waitall(pcl->n_reqs,pcl->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(pcl->n_reqs,pcl->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_res);CHKERRQ(ierr);
  /* sum results of my neighbors */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = VecSet(pcl->vec1,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(pcl->vec1,ft->n_shared_lb[i],ft->shared_lb[i],pcl->work_vecs[i-1],INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(pcl->vec1);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(pcl->vec1);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,pcl->vec1);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "PCApplyLocalWithPolling_LUMPED"
static PetscErrorCode PCApplyLocalWithPolling_LUMPED(PC pc,Vec x,Vec y,PetscInt *_n2c)
{
  PCFT_LUMPED         *pcl = (PCFT_LUMPED*)pc->data;
  PetscErrorCode      ierr;
  FETI                ft   = pcl->ft;
  Subdomain           sd   = ft->subdomain;
  PetscMPIInt         i_mpi;
  PetscInt            i,is,ir;
  Vec                 vec,vec_res,vec_aux;
  PetscInt            n2c,n2c0;
  const PetscScalar   *array_s;
  
  PetscFunctionBegin;
  /* allocate resources if not available */
  if(!pcl->work_vecs) { ierr = PCAllocateFETIWorkVecs_Private(pc,ft);CHKERRQ(ierr);}

  n2c  = ((!x)==0);
  n2c0 = n2c;
  ierr = PCAllocateCommunication_Private(pc,&n2c);CHKERRQ(ierr);
  if (_n2c) *_n2c = n2c;

  if (n2c) {
    is = 0;
    if (n2c0) {
      /* Application of B_Ddelta^T */
      ierr = MatMultTranspose(ft->B_Ddelta,x,sd->vec1_B);CHKERRQ(ierr);
      /* Application of local A_BB */
      ierr = MatMult(pcl->A_BB,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
      /* Application of B_Ddelta */
      ierr = MatMult(ft->B_Ddelta,sd->vec2_B,y);CHKERRQ(ierr);

      /* send to my neighbors my local vector to which my neighbors' preconditioner must be applied */
      for (is=0; is<ft->n_neigh_lb-1; is++){
	ierr = VecGetSubVector(x,pcl->isindex[is],&vec);CHKERRQ(ierr);
	ierr = VecGetArrayRead(vec,&array_s);CHKERRQ(ierr);   
	ierr = PetscMPIIntCast(ft->neigh_lb[is+1],&i_mpi);CHKERRQ(ierr);   
	ierr = MPI_Isend(array_s,ft->n_shared_lb[is+1],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->s_reqs[is]);CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(vec,&array_s);CHKERRQ(ierr);   
	ierr = VecRestoreSubVector(x,pcl->isindex[is],&vec);CHKERRQ(ierr);
      }
    }
    /* receive vectors from my neighbors */
    ir = 0;
    for (i=0; i<ft->n_neigh_lb-1; i++){
      if (pcl->pnc[i]) {
	ierr = PetscMPIIntCast(ft->neigh_lb[i+1],&i_mpi);CHKERRQ(ierr);
	ierr = MPI_Irecv(pcl->work_vecs[i],ft->n_shared_lb[i+1],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->r_reqs[ir++]);CHKERRQ(ierr);
      }
    }
    ierr = MPI_Waitall(ir,pcl->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    if(is) { ierr = MPI_Waitall(is,pcl->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}

    /* apply preconditioner to received vectors */
    ierr = VecDuplicate(pcl->vec1,&vec_res);CHKERRQ(ierr);
    is   = 0;
    for (i=0; i<ft->n_neigh_lb-1; i++){
      if (pcl->pnc[i]) {
	ierr = VecSet(pcl->vec1,0.0);CHKERRQ(ierr);
	ierr = VecSetValues(pcl->vec1,ft->n_shared_lb[i+1],ft->shared_lb[i+1],pcl->work_vecs[i],INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(pcl->vec1);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(pcl->vec1);CHKERRQ(ierr);
	/* Application of B_Ddelta^T */
	ierr = MatMultTranspose(ft->B_Ddelta,pcl->vec1,sd->vec1_B);CHKERRQ(ierr);
	/* Application of local A_BB */
	ierr = MatMult(pcl->A_BB,sd->vec1_B,sd->vec2_B);CHKERRQ(ierr);
	/* Application of B_Ddelta */
	ierr = MatMult(ft->B_Ddelta,sd->vec2_B,vec_res);CHKERRQ(ierr);
	/* communicate result */
	ierr = VecGetSubVector(vec_res,pcl->isindex[i],&vec_aux);CHKERRQ(ierr);
	ierr = VecGetArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
	ierr = PetscMPIIntCast(ft->neigh_lb[i+1],&i_mpi);CHKERRQ(ierr);   
	ierr = MPI_Isend(array_s,ft->n_shared_lb[i+1],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->s_reqs[is++]);CHKERRQ(ierr);
	ierr = VecRestoreArrayRead(vec_aux,&array_s);CHKERRQ(ierr);   
	ierr = VecRestoreSubVector(vec_res,pcl->isindex[i],&vec_aux);CHKERRQ(ierr);
      }
    } 
    /* receive results */
    ir = 0;
    if (n2c0) {
      for (ir=0; ir<ft->n_neigh_lb-1; ir++){
	ierr = PetscMPIIntCast(ft->neigh_lb[ir+1],&i_mpi);CHKERRQ(ierr);
	ierr = MPI_Irecv(pcl->work_vecs[ir],ft->n_shared_lb[ir+1],MPIU_SCALAR,i_mpi,pcl->tag,pcl->comm,&pcl->r_reqs[ir]);CHKERRQ(ierr);    
      }
    }
    if (ir) { ierr = MPI_Waitall(ir,pcl->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
    ierr = MPI_Waitall(is,pcl->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_res);CHKERRQ(ierr);

    if(n2c0) {
      /* sum results of my neighbors */
      for (i=0; i<ft->n_neigh_lb-1; i++){
	ierr = VecSet(pcl->vec1,0.0);CHKERRQ(ierr);
	ierr = VecSetValues(pcl->vec1,ft->n_shared_lb[i+1],ft->shared_lb[i+1],pcl->work_vecs[i],INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(pcl->vec1);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(pcl->vec1);CHKERRQ(ierr);
	ierr = VecAXPY(y,1.0,pcl->vec1);CHKERRQ(ierr);
      }
    }
  }
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

    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocalWithPolling_C",PCApplyLocalWithPolling_LUMPED);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCApplyLocal_C",PCApplyLocal_LUMPED);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}
EXTERN_C_END
