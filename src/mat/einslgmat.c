#include <einsmat.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <private/einsmatimpl.h>


#undef __FUNCT__
#define __FUNCT__ "LGMatDestroyMat_Private"
static PetscErrorCode LGMatDestroyMat_Private(Mat A)
{
  LGMat_ctx    mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellUnAsmGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->localA);CHKERRQ(ierr);
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "LGMatMult_Private"
static PetscErrorCode LGMatMult_Private(Mat A,Vec x,Vec y)
{
  LGMat_ctx      mat_ctx;
  PetscErrorCode ierr;
  PetscBool      flg;
  Vec            xin,yin;
  
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)x,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot MatMult non globally unassembled vector");
  ierr = PetscObjectTypeCompare((PetscObject)y,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)y),PETSC_ERR_SUP,"Cannot save result of MatMult operation in non globally unassembled vector");

  ierr = VecUnAsmGetLocalVectorRead(x,&xin);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(y,&yin);CHKERRQ(ierr);
  ierr = MatShellUnAsmGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = MatMult(mat_ctx->localA,xin,yin);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(y,yin);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(x,xin);CHKERRQ(ierr);
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateLGMat"
/*@
   MatCreateLGMat - Creates a MATSHELLUNASM matrix with no associated
   operations. This matrix is mainly intended for associating a global
   communicator to a locally defined matrix. The local matrix is part
   of the context of the MATSHELLUNASM matrix, and it is accessed as
   ctx->localA.

   Input Parameter:
+  comm            - the MPI communicator
.  m,n             - size of the local matrix
.  localA          - the local matrix. It can be NULL, in which case is not set

   Output Parameter:
+  A               - the created matrix

   Level: intermediate

#@*/
PETSC_EXTERN PetscErrorCode MatCreateLGMat(MPI_Comm comm,PetscInt m,PetscInt n,Mat localA,Mat *A)
{
  PetscErrorCode   ierr;
  LGMat_ctx        matctx;

  PetscFunctionBegin;
  /* creating the mat context for the MatShell*/
  ierr = PetscNew(&matctx);CHKERRQ(ierr);
  /* creating the MatShell */
  ierr = MatCreateShellUnAsm(comm,m,n,m,n,matctx,A);CHKERRQ(ierr);
  ierr = MatShellUnAsmSetOperation(*A,MATOP_DESTROY,(void (*)(void)) LGMatDestroyMat_Private);CHKERRQ(ierr);
  ierr = MatShellUnAsmSetOperation(*A,MATOP_MULT,(void (*)(void)) LGMatMult_Private);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  if (localA) {
    matctx->localA = localA;
    ierr = PetscObjectReference((PetscObject)localA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
