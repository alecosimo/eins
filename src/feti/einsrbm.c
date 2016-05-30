#include <../src/feti/einsrbm.h>


#undef __FUNCT__
#define __FUNCT__ "FETICSRBMSetStiffnessMatrixFunction"
/*@C
   FETICSRBMSetStiffnessMatrixFunction - Set the function to compute the stiffness matrix.

   Logically Collective on FETICS

   Input Parameters:
+  ft   - the FETI context 
.  S    - matrix to hold the stiffness matrix (or NULL to have it created internally)
.  fun  - the function evaluation routine
-  ctx  - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of fun:
$  fun(FETICS ftcs,Mat stiffness,ctx);

+  ftcs      - FETICS context 
.  stiffness - The matrix to hold the stiffness matrix
-  ctx       - [optional] user-defined context for matrix evaluation routine (may be NULL)

   Level: beginner

.keywords: FETICS, stiffness matrix, rigid body modes

@*/
PETSC_EXTERN PetscErrorCode FETICSRBMSetStiffnessMatrixFunction(FETI ft,Mat S,FETICSRBMIStiffness fun,void *ctx)
{
  RBM_CS         *gn;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  if (!ft->ftcs) {
    ierr = FETICSCreate(PetscObjectComm((PetscObject)ft),&ft->ftcs);CHKERRQ(ierr);
    ierr = FETICSSetType(ft->ftcs,ft,ft->ftcs_type);CHKERRQ(ierr);
    ierr = FETICSSetFromOptions(ft->ftcs);CHKERRQ(ierr);
  }
  gn = (RBM_CS*)(ft->ftcs)->data;
  if (S) {
    PetscValidHeaderSpecific(S,MAT_CLASSID,2);
    ierr = PetscObjectReference((PetscObject)S);CHKERRQ(ierr);
  }
  gn->stiffnessFun  = fun;
  gn->stiffness_mat = S;
  gn->stiffness_ctx = ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSComputeCoarseBasis_RBM"
static PetscErrorCode FETICSComputeCoarseBasis_RBM(FETICS ftcs,Mat *localG,Mat *R)
{
  PetscErrorCode    ierr;
  RBM_CS            *gn = (RBM_CS*)ftcs->data;
  
  PetscFunctionBegin;
  if (gn->state>=RBM_STATE_COMPUTED) PetscFunctionReturn(0);
  if (gn->localG) {
    ierr = MatDestroy(localG);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)gn->localG);CHKERRQ(ierr);
  }
  *localG   = gn->localG;
  if (R && gn->rbm) {
    ierr = MatDestroy(R);CHKERRQ(ierr);
    *R   = gn->rbm;
    ierr = PetscObjectReference((PetscObject)gn->rbm);CHKERRQ(ierr);
  }
  gn->state = RBM_STATE_COMPUTED;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSSetUp_RBM"
static PetscErrorCode FETICSSetUp_RBM(FETICS ftcs)
{
  PetscErrorCode   ierr;
  FETI             ft = ftcs->feti;
  RBM_CS           *gn = (RBM_CS*)ftcs->data;
  MPI_Comm         comm;
  Subdomain        sd = ft->subdomain;
  Mat              x;
  PC               pc;
  PetscBool        issbaij;
  
  PetscFunctionBegin;
  ierr   = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  if (!gn->stiffnessFun) {
    /* get number of rigid body modes */
    ierr   = MatMumpsGetInfog(ft->F_neumann,28,&ft->n_cs);CHKERRQ(ierr);
    if(ft->n_cs){
      /* Compute rigid body modes */
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,ft->n_cs,NULL,&gn->rbm);CHKERRQ(ierr);
      ierr = MatDuplicate(gn->rbm,MAT_DO_NOT_COPY_VALUES,&x);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(gn->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(gn->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatMumpsSetIcntl(ft->F_neumann,25,-1);CHKERRQ(ierr);
      ierr = MatMatSolve(ft->F_neumann,x,gn->rbm);CHKERRQ(ierr);
      ierr = MatDestroy(&x);CHKERRQ(ierr);

      /* compute matrix localG */
      ierr = MatGetSubMatrix(gn->rbm,sd->is_B_local,NULL,MAT_INITIAL_MATRIX,&x);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,ft->n_cs,NULL,&gn->localG);CHKERRQ(ierr);
      ierr = MatMatMult(ft->B_delta,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&gn->localG);CHKERRQ(ierr);    
    }
  } else {
    /* Solve system and get number of rigid body modes */
    if (!gn->stiffness_mat) {
      ierr = MatDuplicate(sd->localA,MAT_SHARE_NONZERO_PATTERN,&gn->stiffness_mat);CHKERRQ(ierr);
    }
    ierr = (*gn->stiffnessFun)(ftcs,gn->stiffness_mat,gn->stiffness_ctx);CHKERRQ(ierr);
    if (!gn->ksp_rbm) {
      ierr = KSPCreate(PETSC_COMM_SELF,&gn->ksp_rbm);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)gn->ksp_rbm,(PetscObject)ft,1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)ft,(PetscObject)gn->ksp_rbm);CHKERRQ(ierr);
      ierr = KSPSetType(gn->ksp_rbm,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(gn->ksp_rbm,&pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)(gn->stiffness_mat),MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
      if (issbaij) {
	ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
      } else {
	ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
      }
      ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
      ierr = KSPSetOperators(gn->ksp_rbm,gn->stiffness_mat,gn->stiffness_mat);CHKERRQ(ierr);
      /* prefix for setting options */
      ierr = KSPSetOptionsPrefix(gn->ksp_rbm,"fetics_rbm_");CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(gn->stiffness_mat,"fetics_rbm_");CHKERRQ(ierr);
      ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
      ierr = PCFactorGetMatrix(pc,&gn->F_rbm);CHKERRQ(ierr);
      /* sequential ordering */
      ierr = MatMumpsSetIcntl(gn->F_rbm,7,2);CHKERRQ(ierr);
      /* Null row pivot detection */
      ierr = MatMumpsSetIcntl(gn->F_rbm,24,1);CHKERRQ(ierr);
      /* threshhold for row pivot detection */
      ierr = MatMumpsSetCntl(gn->F_rbm,3,1.e-6);CHKERRQ(ierr);

      /* Maybe the following two options should be given as external options and not here*/
      ierr = KSPSetFromOptions(gn->ksp_rbm);CHKERRQ(ierr);
      ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(gn->ksp_rbm,gn->stiffness_mat,gn->stiffness_mat);CHKERRQ(ierr);
    }
    /* Set Up KSP for Neumann problem: here the factorization takes place!!! */
    ierr  = KSPSetUp(gn->ksp_rbm);CHKERRQ(ierr);
    ierr  = MatMumpsGetInfog(gn->F_rbm,28,&ft->n_cs);CHKERRQ(ierr);
    if(ft->n_cs){
      /* Compute rigid body modes */
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,ft->n_cs,NULL,&gn->rbm);CHKERRQ(ierr);
      ierr = MatDuplicate(gn->rbm,MAT_DO_NOT_COPY_VALUES,&x);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(gn->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(gn->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatMumpsSetIcntl(gn->F_rbm,25,-1);CHKERRQ(ierr);
      ierr = MatMatSolve(gn->F_rbm,x,gn->rbm);CHKERRQ(ierr);
      ierr = MatDestroy(&x);CHKERRQ(ierr);

      /* compute matrix localG */
      ierr = MatGetSubMatrix(gn->rbm,sd->is_B_local,NULL,MAT_INITIAL_MATRIX,&x);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,ft->n_cs,NULL,&gn->localG);CHKERRQ(ierr);
      ierr = MatSetOption(gn->localG,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatMatMult(ft->B_delta,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&gn->localG);CHKERRQ(ierr);    
      ierr = MatDestroy(&x);CHKERRQ(ierr);
    }
    ierr = KSPDestroy(&gn->ksp_rbm);CHKERRQ(ierr);
    ierr = MatDestroy(&gn->stiffness_mat);CHKERRQ(ierr);
    ierr = MatDestroy(&gn->rbm);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSDestroy_RBM"
static PetscErrorCode FETICSDestroy_RBM(FETICS ftcs)
{
  PetscErrorCode ierr;
  RBM_CS         *gn = (RBM_CS*)ftcs->data;
  
  PetscFunctionBegin;
  ierr = MatDestroy(&gn->localG);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->rbm);CHKERRQ(ierr);
  ierr = PetscFree(ftcs->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSCreate_RBM"
PETSC_EXTERN PetscErrorCode FETICSCreate_RBM(FETICS);
PetscErrorCode FETICSCreate_RBM(FETICS ftcs)
{
  PetscErrorCode  ierr;
  FETI            ft = ftcs->feti;
  RBM_CS        *gn;
  
  PetscFunctionBegin;
  ierr       = PetscNewLog(ft,&gn);CHKERRQ(ierr);
  ftcs->data = (void*)gn;

  ierr      = PetscMemzero(gn,sizeof(RBM_CS));CHKERRQ(ierr);
  gn->state = RBM_STATE_INITIAL;
    
  ftcs->ops->setup               = FETICSSetUp_RBM;
  ftcs->ops->destroy             = FETICSDestroy_RBM;
  ftcs->ops->setfromoptions      = 0;
  ftcs->ops->computecoarsebasis  = FETICSComputeCoarseBasis_RBM; 
  PetscFunctionReturn(0);
}

