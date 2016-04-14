
/*
   This provides a simple shell for creating a very simple globally
  unassembled matrix class for use with KSP without coding much of
  anything.
*/

#include <einsmat.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>

static PetscErrorCode MatDestroy_ShellUnAsm(Mat);
static PetscErrorCode MatMultAdd_ShellUnAsm(Mat,Vec,Vec,Vec);
static PetscErrorCode MatCreateVecs_MatShellUnAsm(Mat,Vec*,Vec*);
static PetscErrorCode MatMultTransposeAdd_ShellUnAsm(Mat,Vec,Vec,Vec);
static PetscErrorCode LayoutSetUp_Private(PetscLayout);

typedef struct {
  PetscErrorCode (*destroy)(Mat);
  Vec         right_add_work;
  Vec         left_add_work;
  void        *ctx;
} Mat_ShellUnAsm;


#undef __FUNCT__
#define __FUNCT__ "MatCreateVecs_MatShellUnAsm"
/*@
   MatCreateVecs_MatShellUnAsm - Gets globally unassembled vector(s) compatible with the matrix

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameter:
+   right - (optional) vector that the matrix can be multiplied against
-   left - (optional) vector that the matrix vector product can be stored in


  Notes: These are new vectors which are not owned by the Mat, they should be destroyed in VecDestroy() when no longer needed

  Level: advanced

.seealso: MatCreate(), VecDestroy()
@*/
static PetscErrorCode MatCreateVecs_MatShellUnAsm(Mat mat,Vec *right,Vec *left)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot create vectors from non-shell matrix");
  
  if (right) {
    if (mat->cmap->n < 0) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"PetscLayout for columns not yet setup");
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,mat->cmap->N);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECMPIUNASM);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->cmap,&(*right)->map);CHKERRQ(ierr);
  }
  if (left) {
    if (mat->rmap->n < 0) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"PetscLayout for rows not yet setup");
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,mat->rmap->N);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECMPIUNASM);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->rmap,&(*left)->map);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatShellUnAsmGetContext"
/*@
    MatShellUnAsmGetContext - Returns the user-provided context
    associated with a globally unassembled shell matrix.

    Not Collective

    Input Parameter:
.   mat - the matrix, should have been created with MatCreateShell()

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

.keywords: matrix, shell, get, context

.seealso: MatCreateShellUnAsm(), MatShellUnAsmSetOperation(), MatShellUnAsmSetContext()
@*/
PETSC_EXTERN PetscErrorCode MatShellUnAsmGetContext(Mat mat,void *ctx)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(ctx,2);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if (flg) *(void**)ctx = ((Mat_ShellUnAsm*)(mat->data))->ctx;
  else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get context from non-shell matrix");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_ShellUnAsm"
static PetscErrorCode MatDestroy_ShellUnAsm(Mat mat)
{
  PetscErrorCode   ierr;
  Mat_ShellUnAsm   *shell = (Mat_ShellUnAsm*)mat->data;

  PetscFunctionBegin;
  if (shell->destroy) {
    ierr = (*shell->destroy)(mat);CHKERRQ(ierr);
  }
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_ShellUnAsm"
static PetscErrorCode MatMultAdd_ShellUnAsm(Mat A,Vec x,Vec y,Vec z)
{
  Mat_ShellUnAsm  *shell = (Mat_ShellUnAsm*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->right_add_work) {ierr = VecDuplicate(z,&shell->right_add_work);CHKERRQ(ierr);}
    ierr = MatMult(A,x,shell->right_add_work);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,shell->right_add_work);CHKERRQ(ierr);
  } else {
    ierr = MatMult(A,x,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_ShellUnAsm"
static PetscErrorCode MatMultTransposeAdd_ShellUnAsm(Mat A,Vec x,Vec y,Vec z)
{
  Mat_ShellUnAsm  *shell = (Mat_ShellUnAsm*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (y == z) {
    if (!shell->left_add_work) {ierr = VecDuplicate(z,&shell->left_add_work);CHKERRQ(ierr);}
    ierr = MatMultTranspose(A,x,shell->left_add_work);CHKERRQ(ierr);
    ierr = VecWAXPY(z,1.0,shell->left_add_work,y);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(A,x,z);CHKERRQ(ierr);
    ierr = VecAXPY(z,1.0,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "LayoutSetUp_Private"
static PetscErrorCode LayoutSetUp_Private(PetscLayout map)
{
  PetscFunctionBegin;
  if ((map->n >= 0) && (map->N >= 0)) PetscFunctionReturn(0);
  if (map->n > 0 && map->bs > 1) {
    if (map->n % map->bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local matrix size %D must be divisible by blocksize %D",map->n,map->bs);
  }
  if (map->N > 0 && map->bs > 1) {
    if (map->N % map->bs) SETERRQ2(map->comm,PETSC_ERR_PLIB,"Global matrix size %D must be divisible by blocksize %D",map->N,map->bs);
  }  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreate_ShellUnAsm"
/*M
 
  MATSHELLUNASM - MATSHELLUNASM = "shellunasm" - A globally
   unassembled matrix type to be used to define your own matrix type
   -- perhaps matrix free.

  Level: advanced

.seealso: MatCreateShellUnAsm
M*/
PETSC_EXTERN PetscErrorCode MatCreate_ShellUnAsm(Mat A)
{
  Mat_ShellUnAsm  *b;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr    = PetscNewLog(A,&b);CHKERRQ(ierr);
  A->data = (void*)b;
  
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->ops->destroy = MatDestroy_ShellUnAsm;
  A->ops->getvecs = MatCreateVecs_MatShellUnAsm;

  ierr = LayoutSetUp_Private(A->rmap);CHKERRQ(ierr);
  ierr = LayoutSetUp_Private(A->cmap);CHKERRQ(ierr);

  b->ctx            = 0;
  b->destroy        = 0;
  b->right_add_work = 0;
  b->left_add_work  = 0;
  A->assembled      = PETSC_TRUE;
  A->preallocated   = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSHELLUNASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateShellUnAsm"
/*@C 
   MatCreateShellUnAsm - Creates a new globally unassembled matrix
   class for use with a user-defined private data storage format.

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (must be given)
.  n - number of local columns (must be given)
.  M - number of global rows (must be given)
.  N - number of global columns (must be given)
-  ctx - pointer to data needed by the shell matrix routines

   Output Parameter:
.  A - the matrix

   Level: advanced

.keywords: matrix, shell, create

.seealso: MatShellUnAsmSetOperation(), MatHasOperation(), MatShellUnAsmGetContext(), MatShellUnAsmSetContext()
@*/
PETSC_EXTERN PetscErrorCode MatCreateShellUnAsm(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,void *ctx,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (M==PETSC_DETERMINE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Error: size M must be provided, it can not be equal to PETSC_DETERMINE.");
  if (N==PETSC_DETERMINE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Error: size N must be provided, it can not be equal to PETSC_DETERMINE.");
  if (m==PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Error: size m must be provided, it can not be equal to PETSC_DECIDE.");
  if (n==PETSC_DECIDE) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Error: size n must be provided, it can not be equal to PETSC_DECIDE.");

  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSHELLUNASM);CHKERRQ(ierr);
  ierr = MatShellUnAsmSetContext(*A,ctx);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatShellUnAsmSetContext"
/*@
    MatShellUnAsmContext - sets the context for a globally unassmebled shell matrix

   Logically Collective on Mat

    Input Parameters:
+   mat - the shell matrix
-   ctx - the context

   Level: advanced

.seealso: MatCreateShellUnAsm(), MatShellUnAsmGetContext(), MatShellUnAsmGetOperation()
@*/
PETSC_EXTERN PetscErrorCode  MatShellUnAsmSetContext(Mat mat,void *ctx)
{
  Mat_ShellUnAsm  *shell = (Mat_ShellUnAsm*)mat->data;
  PetscErrorCode  ierr;
  PetscBool       flg; 

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot attach context to non-shell matrix");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatShellUnAsmSetOperation"
/*@C
    MatShellUnAsmSetOperation - Allows user to set a matrix operation for
                              a globally unassembled shell matrix.

   Logically Collective on Mat

    Input Parameters:
+   mat - the shell matrix
.   op - the name of the operation
-   f - the function that provides the operation.

   Level: advanced

.keywords: matrix, shell, set, operation

.seealso: MatCreateShellUnAsm(), MatShellUnAsmGetContext(), MatShellUnAsmGetOperation(), MatShellUnAsmSetContext()
@*/
PETSC_EXTERN PetscErrorCode MatShellUnAsmSetOperation(Mat mat,MatOperation op,void (*f)(void))
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  switch (op) {
  case MATOP_DESTROY:
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
    if (flg) {
      Mat_ShellUnAsm *shell = (Mat_ShellUnAsm*)mat->data;
      shell->destroy = (PetscErrorCode (*)(Mat))f;
    } else mat->ops->destroy = (PetscErrorCode (*)(Mat))f;
    break;
  case MATOP_VIEW:
    mat->ops->view = (PetscErrorCode (*)(Mat,PetscViewer))f;
    break;
  case MATOP_MULT:
    mat->ops->mult = (PetscErrorCode (*)(Mat,Vec,Vec))f;
    if (!mat->ops->multadd) mat->ops->multadd = MatMultAdd_ShellUnAsm;
    break;
  case MATOP_MULT_TRANSPOSE:
    mat->ops->multtranspose = (PetscErrorCode (*)(Mat,Vec,Vec))f;
    if (!mat->ops->multtransposeadd) mat->ops->multtransposeadd = MatMultTransposeAdd_ShellUnAsm;
    break;
  case MATOP_GET_VECS:
    mat->ops->getvecs = (PetscErrorCode (*)(Mat,Vec*,Vec*))f;
    break;
  default:
    (((void(**)(void))mat->ops)[op]) = f;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatShellUnAsmGetOperation"
/*@C
    MatShellUnAsmGetOperation - Gets a matrix function for a globally unassembled shell matrix.

    Not Collective

    Input Parameters:
+   mat - the shell matrix
-   op - the name of the operation

    Output Parameter:
.   f - the function that provides the operation.

    Level: advanced

.keywords: matrix, shell, set, operation

.seealso: MatCreateShellUnAsm(), MatShellUnAsmGetContext(), MatShellUnAsmSetOperation(), MatShellUnAsmSetContext()
@*/
PETSC_EXTERN PetscErrorCode MatShellUnAsmGetOperation(Mat mat,MatOperation op,void(**f)(void))
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  if (op == MATOP_DESTROY) {
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
    if (flg) {
      Mat_ShellUnAsm *shell = (Mat_ShellUnAsm*)mat->data;
      *f = (void (*)(void))shell->destroy;
    } else {
      *f = (void (*)(void))mat->ops->destroy;
    }
  } else if (op == MATOP_VIEW) {
    *f = (void (*)(void))mat->ops->view;
  } else {
    *f = (((void (**)(void))mat->ops)[op]);
  }
  PetscFunctionReturn(0);
}


