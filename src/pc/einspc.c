#include <private/einspcimpl.h>
#include <private/einsfetiimpl.h>


#undef __FUNCT__
#define __FUNCT__ "PCApplyLocal"
/*@
   PCApplyLocal - Sets the variable controlling the computation of the jacobian.

   Input Parameters:
.  ksp - the KSP context
.  flg - flag to set

   Level: intermediate

@*/
PETSC_EXTERN PetscErrorCode PCApplyLocal(PC pc,Vec x, Vec y) {
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ierr = PetscUseMethod(pc,"PCApplyLocal_C",(PC,Vec,Vec),(pc,x,y));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCAllocateFETIWorkVecs_Private"
PetscErrorCode PCAllocateFETIWorkVecs_Private(PC pc, FETI ft) {
  PetscErrorCode ierr;
  PCFT_BASE      *pch = (PCFT_BASE*)pc->data;
  PetscInt       i,total;
  
  PetscFunctionBegin;
  pch->n_reqs = ft->n_neigh_lb - 1;
  total       = 0;
  for (i=1;i<ft->n_neigh_lb;i++) total += ft->n_shared_lb[i];
  ierr = PetscMalloc1(pch->n_reqs,&pch->r_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pch->n_reqs,&pch->s_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pch->n_reqs,&pch->work_vecs);CHKERRQ(ierr);
  ierr = PetscMalloc1(total,&pch->work_vecs[0]);CHKERRQ(ierr);
  for (i=1;i<pch->n_reqs;i++) pch->work_vecs[i] = pch->work_vecs[i-1]+ft->n_shared_lb[i];
  ierr = PetscMalloc1(pch->n_reqs,&pch->isindex);CHKERRQ(ierr);
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&pch->isindex[i-1]);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm((PetscObject)ft,&pch->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCDeAllocateFETIWorkVecs_Private"
PetscErrorCode PCDeAllocateFETIWorkVecs_Private(PC pc) {
  PetscErrorCode ierr;
  PCFT_BASE      *pch = (PCFT_BASE*)pc->data;
  PetscInt       i;
  
  PetscFunctionBegin;
  ierr = PetscFree(pch->s_reqs);CHKERRQ(ierr);
  ierr = PetscFree(pch->r_reqs);CHKERRQ(ierr);
  ierr = PetscFree(pch->work_vecs[0]);CHKERRQ(ierr);
  ierr = PetscFree(pch->work_vecs);CHKERRQ(ierr);
  for (i=0;i<pch->n_reqs;i++){ ierr = ISDestroy(&pch->isindex[i]);CHKERRQ(ierr);}
  ierr = PetscFree(pch->isindex);CHKERRQ(ierr);
  pch->n_reqs = 0;
  PetscFunctionReturn(0);
}
