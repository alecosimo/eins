#include <private/einspcimpl.h>
#include <private/einsfetiimpl.h>


#undef __FUNCT__
#define __FUNCT__ "PCApplyLocal"
/*@
   PCApplyLocal - Applies the preconditioner locally.

   Input Parameters:
.  PC - the pc context
.  x - the vector to which to apply the preconditioner (it can be NULL)

   Output Parameters:
.  y   - result of the application of the preconditioner (it can be NULL)
.  n2c - "needs to communicate": it means that there are proceesses that need to communicate to the current process

   Level: developer

@*/
PETSC_EXTERN PetscErrorCode PCApplyLocal(PC pc,Vec x,Vec y,PetscInt *n2c)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if(x) { PetscValidHeaderSpecific(x,VEC_CLASSID,2);}
  if(y) { PetscValidHeaderSpecific(y,VEC_CLASSID,3);}
  ierr = PetscUseMethod(pc,"PCApplyLocal_C",(PC,Vec,Vec,PetscInt*),(pc,x,y,n2c));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCApplyLocalMult"
/*@
   PCApplyLocalMult - Applies the preconditioner locally to multiple vectors.

   Input Parameters:
.  PC - the pc context
.  X  - the matrix with column vectors to which to apply the preconditioner

   Output Parameters:
.  Y   - matrix to store the results

   Level: developer

@*/
PETSC_EXTERN PetscErrorCode PCApplyLocalMult(PC pc,Mat X,Mat Y)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(X,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Y,MAT_CLASSID,3);
  ierr = PetscUseMethod(pc,"PCApplyLocalMult_C",(PC,Mat,Mat),(pc,X,Y));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCAllocateFETIWorkVecs_Private"
PetscErrorCode PCAllocateFETIWorkVecs_Private(PC pc, FETI ft)
{
  PetscErrorCode ierr;
  PCFT_BASE      *pch = (PCFT_BASE*)pc->data;
  PetscInt       i,total;
  Vec            lambda_local;
  
  PetscFunctionBegin;
  pch->n_reqs = ft->n_neigh_lb - 1;
  total       = 0;
  for (i=1;i<ft->n_neigh_lb;i++) total += ft->n_shared_lb[i];
  ierr = PetscMalloc1(pch->n_reqs,&pch->r_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pch->n_reqs,&pch->s_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pch->n_reqs,&pch->work_vecs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pch->n_reqs,&pch->pnc);CHKERRQ(ierr);
  ierr = PetscMalloc1(total,&pch->work_vecs[0]);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVectorRead(ft->lambda_global,&lambda_local);CHKERRQ(ierr);
  ierr = VecDuplicate(lambda_local,&pch->vec1);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(ft->lambda_global,lambda_local);CHKERRQ(ierr);
  for (i=1;i<pch->n_reqs;i++) pch->work_vecs[i] = pch->work_vecs[i-1]+ft->n_shared_lb[i];
  ierr = PetscMalloc1(pch->n_reqs,&pch->isindex);CHKERRQ(ierr);
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&pch->isindex[i-1]);CHKERRQ(ierr);
  }
  /* this communicator is going to be used by an external library */
  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)ft),&pch->comm);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCDeAllocateFETIWorkVecs_Private"
PetscErrorCode PCDeAllocateFETIWorkVecs_Private(PC pc)
{
  PetscErrorCode ierr;
  PCFT_BASE      *pch = (PCFT_BASE*)pc->data;
  PetscInt       i;
  
  PetscFunctionBegin;
  ierr = PetscFree(pch->s_reqs);CHKERRQ(ierr);
  ierr = PetscFree(pch->r_reqs);CHKERRQ(ierr);
  ierr = PetscFree(pch->pnc);CHKERRQ(ierr);
  ierr = PetscFree(pch->work_vecs[0]);CHKERRQ(ierr);
  ierr = PetscFree(pch->work_vecs);CHKERRQ(ierr);
  ierr = VecDestroy(&pch->vec1);CHKERRQ(ierr);
  for (i=0;i<pch->n_reqs;i++){ ierr = ISDestroy(&pch->isindex[i]);CHKERRQ(ierr);}
  ierr = PetscFree(pch->isindex);CHKERRQ(ierr);
  ierr = MPI_Comm_free(&pch->comm);CHKERRQ(ierr);
  pch->n_reqs = 0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCAllocateCommunication_Private"
PetscErrorCode PCAllocateCommunication_Private(PC pc,PetscInt *n2c)
{
  PetscErrorCode ierr;
  PCFT_BASE      *pch = (PCFT_BASE*)pc->data;
  PetscInt       i;
  FETI           ft   = pch->ft;
  PetscMPIInt    i_mpi;
  
  PetscFunctionBegin;
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
    ierr = MPI_Isend(n2c,1,MPIU_INT,i_mpi,10,pch->comm,&pch->s_reqs[i-1]);CHKERRQ(ierr);
    ierr = MPI_Irecv(&pch->pnc[i-1],1,MPIU_INT,i_mpi,10,pch->comm,&pch->r_reqs[i-1]);CHKERRQ(ierr);    
  }
  ierr = MPI_Waitall(pch->n_reqs,pch->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(pch->n_reqs,pch->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,n2c,1,MPIU_INT,MPI_SUM,pch->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
