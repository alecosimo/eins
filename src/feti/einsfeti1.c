#include <../src/feti/einsfeti1.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einssys.h>


/* private functions*/
static PetscErrorCode FETI1SetUpNeumannSolver_Private(FETI);
static PetscErrorCode FETI1ComputeMatrixG_Private(FETI);
static PetscErrorCode FETI1ComputeRhsE_Private(FETI);
static PetscErrorCode FETI1BuildInterfaceProblem_Private(FETI);
static PetscErrorCode FETIDestroy_FETI1(FETI);
static PetscErrorCode FETISetUp_FETI1(FETI);
static PetscErrorCode FETI1DestroyMatF_Private(Mat);
static PetscErrorCode FETI1MatMult_Private(Mat,Vec,Vec);
static PetscErrorCode FETISetFromOptions_FETI1(PetscOptionItems*,FETI);
static PetscErrorCode FETI1SetUpCoarseProblem_Private(FETI);
static PetscErrorCode FETI1FactorizeCoarseProblem_Private(FETI);
static PetscErrorCode FETI1ApplyCoarseProblem_Private(FETI,Vec,Vec);
static PetscErrorCode FETI1ComputeInitialCondition_Private(FETI);
static PetscErrorCode FETI1ComputeAlpha_Private(FETI);
static PetscErrorCode FETISolve_FETI1(FETI,Vec);
static PetscErrorCode FETI1SetInterfaceProblemRHS_Private(FETI);
  
PetscErrorCode FETI1Project_RBM(void*,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI1"
/*@
   FETIDestroy_FETI1 - Destroys the FETI-1 context

   Input Parameters:
.  ft - the FETI context

.seealso FETICreate_FETI1
@*/
static PetscErrorCode FETIDestroy_FETI1(FETI ft)
{
  PetscErrorCode ierr;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  PetscInt       i;
  
  PetscFunctionBegin;
  if (!ft1) PetscFunctionReturn(0);
  ierr = VecDestroy(&ft1->alpha_local);CHKERRQ(ierr);
  ierr = MatDestroy(&ft1->localG);CHKERRQ(ierr);
  ierr = MatDestroy(&ft1->rbm);CHKERRQ(ierr);
  ierr = VecDestroy(&ft1->local_e);CHKERRQ(ierr);
  ierr = KSPDestroy(&ft1->ksp_coarse);CHKERRQ(ierr);
  if(ft1->neigh_holder) {
    ierr = PetscFree(ft1->neigh_holder[0]);CHKERRQ(ierr);
    ierr = PetscFree(ft1->neigh_holder);CHKERRQ(ierr);
  }
  if(ft1->displ) { ierr = PetscFree(ft1->displ);CHKERRQ(ierr);}
  if(ft1->count_rbm) { ierr = PetscFree(ft1->count_rbm);CHKERRQ(ierr);}
  if(ft1->displ_f) { ierr = PetscFree(ft1->displ_f);CHKERRQ(ierr);}
  if(ft1->count_f_rbm) { ierr = PetscFree(ft1->count_f_rbm);CHKERRQ(ierr);}
  for (i=0;i<ft1->n_Gholder;i++) {
    ierr = MatDestroy(&ft1->Gholder[i]);CHKERRQ(ierr);
  }
  if(ft1->coarse_problem) {ierr = MatDestroy(&ft1->coarse_problem);CHKERRQ(ierr);}
  ierr = PetscFree(ft1->Gholder);CHKERRQ(ierr);
  if(ft1->matrices) { ierr = PetscFree(ft1->matrices);CHKERRQ(ierr);}
  ierr = PetscFree(ft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETI1"
/*@
   FETISetUp_FETI1 - Prepares the structures needed by the FETI-1 solver.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETISetUp_FETI1(FETI ft)
{
  PetscErrorCode    ierr;   
  Subdomain         sd = ft->subdomain;
  PetscObjectState  mat_state;

  PetscFunctionBegin;
  if (ft->state==FETI_STATE_INITIAL) {
    ierr = FETIScalingSetUp(ft);CHKERRQ(ierr);
    ierr = FETIBuildLambdaAndB(ft);CHKERRQ(ierr);
    ierr = FETI1SetUpNeumannSolver_Private(ft);CHKERRQ(ierr);
    ierr = FETI1ComputeMatrixG_Private(ft);CHKERRQ(ierr);
    ierr = FETI1ComputeRhsE_Private(ft);CHKERRQ(ierr);
    ierr = FETI1BuildInterfaceProblem_Private(ft);CHKERRQ(ierr);
    ierr = FETI1SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);
    ierr = FETIBuildInterfaceKSP(ft);CHKERRQ(ierr);
    /* set projection in ksp */
    ierr = KSPSetProjection(ft->ksp_interface,FETI1Project_RBM,(void*)ft);CHKERRQ(ierr);
    ierr = KSPSetReProjection(ft->ksp_interface,FETI1Project_RBM,(void*)ft);CHKERRQ(ierr);
    ierr = FETI1SetUpCoarseProblem_Private(ft);CHKERRQ(ierr);
    ierr = FETI1FactorizeCoarseProblem_Private(ft);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectStateGet((PetscObject)sd->localA,&mat_state);CHKERRQ(ierr);
    if (mat_state>ft->mat_state) {
      ierr = PetscObjectStateSet((PetscObject)ft->F,mat_state);CHKERRQ(ierr);  
      if (ft->resetup_pc_interface) {
	PC pc;
	ierr = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
	ierr = PCSetUp(pc);CHKERRQ(ierr);
      }
      ierr = FETI1SetUpNeumannSolver_Private(ft);CHKERRQ(ierr);
      ft->mat_state = mat_state;
    }
    ierr = FETI1ComputeRhsE_Private(ft);CHKERRQ(ierr);
    ierr = FETI1SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);
  }

  ierr = FETI1ComputeInitialCondition_Private(ft);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISetFromOptions_FETI1"
/*@
   FETISetFromOptions_FETI1 - Function to set up options from command line.

   Input Parameter:
.  ft - the FETI context

   Level: beginner

.keywords: FETI, options
@*/
static PetscErrorCode FETISetFromOptions_FETI1(PetscOptionItems *PetscOptionsObject,FETI ft)
{
  PetscErrorCode ierr;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"FETI1 options");CHKERRQ(ierr);

  /* Primal space cumstomization */
  ierr = PetscOptionsBool("-feti1_destroy_coarse","If set, the matrix of the coarse problem, that is (G^T*G) or (G^T*Q*G), will be destroyed","none",ft1->destroy_coarse,&ft1->destroy_coarse,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "FETICreate_FETI1"
/*@
   FETI1 - Implementation of the FETI-1 method. Some comments about options can be put here!

   Options database:
.  -feti_fullyredundant: use fully redundant Lagrange multipliers.
.  -feti_interface_<ksp or pc option>: options for the KSP for the interface problem
.  -feti1_neumann_<ksp or pc option>: for setting pc and ksp options for the neumann solver. 
.  -feti_pc_dirichilet_<ksp or pc option>: options for the KSP or PC to use for solving the Dirichlet problem
   associated to the Dirichlet preconditioner
.  -feti_scaling_type - Sets the scaling type
.  -feti_scaling_factor - Sets a scaling factor different from one
.  -feti1_destroy_coarse - If set, the matrix of the coarse problem, that is (G^T*G) or (G^T*Q*G), will be destroyed after factorization.
.  -feti1_pc_coarse_<ksp or pc option>: options for the KSP for the coarse problem

   Level: beginner

.keywords: FETI, FETI-1
@*/
PetscErrorCode FETICreate_FETI1(FETI ft);
PetscErrorCode FETICreate_FETI1(FETI ft)
{
  PetscErrorCode      ierr;
  FETI_1*             feti1;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ft,&feti1);CHKERRQ(ierr);
  ft->data  = (void*)feti1;

  feti1->res_interface         = 0;
  feti1->alpha_local           = 0;
  feti1->rbm                   = 0;
  feti1->localG                = 0;
  feti1->Gholder               = 0;
  feti1->neigh_holder          = 0;
  feti1->matrices              = 0;
  feti1->n_Gholder             = 0;
  feti1->local_e               = 0;
  feti1->coarse_problem        = 0;
  feti1->F_coarse              = 0;
  feti1->destroy_coarse        = PETSC_FALSE;
  ft->n_cs                     = 0;
  feti1->total_rbm             = 0;
  feti1->max_n_cs             = 0;
  feti1->displ                 = 0;
  feti1->count_rbm             = 0;
  feti1->displ_f               = 0;
  feti1->count_f_rbm           = 0;
  
  /* function pointers */
  ft->ops->setup               = FETISetUp_FETI1;
  ft->ops->destroy             = FETIDestroy_FETI1;
  ft->ops->setfromoptions      = FETISetFromOptions_FETI1;
  ft->ops->computesolution     = FETISolve_FETI1;
  ft->ops->view                = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "FETI1DestroyMatF_Private"
/*@
  FETI1DestroyMatF_Private - Destroy function for the MatShell matrix defining the interface problem's matrix F

   Input Parameters:
.  A - the Matrix context

   Level: developer

.seealso FETI1BuildInterfaceProblem_Private
@*/
static PetscErrorCode FETI1DestroyMatF_Private(Mat A)
{
  FETIMat_ctx    mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellUnAsmGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1MatMult_Private"
/*@
  FETI1MatMult_Private - MatMult function for the MatShell matrix defining the interface problem's matrix F. 
  It performes the product y=F*lambda_global

   Input Parameters:
.  F             - the Matrix context
.  lambda_global - vector to be multiplied by the matrix
.  y             - vector where to save the result of the multiplication

   Level: developer

.seealso FETI1BuildInterfaceProblem_Private
@*/
static PetscErrorCode FETI1MatMult_Private(Mat F, Vec lambda_global, Vec y) /* y=F*lambda_global */
{
  FETIMat_ctx  mat_ctx;
  FETI         ft;
  Subdomain    sd;
  Vec          lambda_local,y_local;
  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellUnAsmGetContext(F,(void**)&mat_ctx);CHKERRQ(ierr);
  ft   = mat_ctx->ft;
  sd   = ft->subdomain;
  ierr = VecUnAsmGetLocalVectorRead(lambda_global,&lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(y,&y_local);CHKERRQ(ierr);
  /* Application of B_delta^T */
  ierr = MatMultTranspose(ft->B_delta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* Application of the already factorized pseudo-inverse */
  ierr = MatMumpsSetIcntl(ft->F_neumann,25,0);CHKERRQ(ierr);
  ierr = MatSolve(ft->F_neumann,sd->vec1_N,sd->vec2_N);CHKERRQ(ierr);
  /* Application of B_delta */
  ierr = VecScatterBegin(sd->N_to_B,sd->vec2_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec2_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(ft->B_delta,sd->vec1_B,y_local);CHKERRQ(ierr);
  /** Communication with other processes is performed for the following operation */
  ierr = VecExchangeBegin(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(ft->exchange_lambda,y,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(lambda_global,lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(y,y_local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1MatGetVecs_Private"
static PetscErrorCode FETI1MatGetVecs_Private(Mat mat,Vec *right,Vec *left)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  FETI           ft  = NULL;
  FETIMat_ctx    mat_ctx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot create vectors from non-shell matrix");
  ierr = MatShellUnAsmGetContext(mat,(void**)&mat_ctx);CHKERRQ(ierr);
  ft   = mat_ctx->ft;
  if (!ft) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix F is missing the FETI context");

  if (right) {
    if (mat->cmap->n < 0) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"PetscLayout for columns not yet setup");
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,mat->cmap->N);CHKERRQ(ierr);
    ierr = VecSetType(*right,VECMPIUNASM);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->cmap,&(*right)->map);CHKERRQ(ierr);
    if(ft->multiplicity) {ierr = VecUnAsmSetMultiplicity(*right,ft->multiplicity);CHKERRQ(ierr);}
  }
  if (left) {
    if (mat->rmap->n < 0) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONGSTATE,"PetscLayout for rows not yet setup");
    ierr = VecCreate(PetscObjectComm((PetscObject)mat),left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,mat->rmap->N);CHKERRQ(ierr);
    ierr = VecSetType(*left,VECMPIUNASM);CHKERRQ(ierr);
    ierr = PetscLayoutReference(mat->rmap,&(*left)->map);CHKERRQ(ierr);
    if(ft->multiplicity) {ierr = VecUnAsmSetMultiplicity(*left,ft->multiplicity);CHKERRQ(ierr);}
  }
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1SetInterfaceProblemRHS_Private"
/*@
   FETI1SetInterfaceProblemRHS_Private - Sets the RHS vector (vector d) of the interface problem.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI1SetInterfaceProblemRHS_Private(FETI ft)
{
  PetscErrorCode ierr;
  Subdomain      sd = ft->subdomain;
  Vec            d_local;
  
  PetscFunctionBegin;
  /** Application of the already factorized pseudo-inverse */
  ierr = MatMumpsSetIcntl(ft->F_neumann,25,0);CHKERRQ(ierr);
  ierr = MatSolve(ft->F_neumann,sd->localRHS,sd->vec1_N);CHKERRQ(ierr);
  /** Application of B_delta */
  ierr = VecUnAsmGetLocalVector(ft->d,&d_local);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(ft->B_delta,sd->vec1_B,d_local);CHKERRQ(ierr);
  /*** Communication with other processes is performed for the following operation */
  ierr = VecExchangeBegin(ft->exchange_lambda,ft->d,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(ft->exchange_lambda,ft->d,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(ft->d,d_local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1BuildInterfaceProblem_Private"
/*@
   FETI1BuildInterfaceProblem_Private - Builds the interface problem, that is the matrix F and the vector d.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI1BuildInterfaceProblem_Private(FETI ft)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  /* Create the MatShell for F */
  ierr = FETICreateFMat(ft,(void (*)(void))FETI1MatMult_Private,(void (*)(void))FETI1DestroyMatF_Private,(void (*)(void))FETI1MatGetVecs_Private);CHKERRQ(ierr);
  /* Creating vector d for the interface problem */
  ierr = MatCreateVecs(ft->F,NULL,&ft->d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1ComputeRhsE_Private"
/*@
   FETI1ComputeRhsE_Private - Computes the rhs term e=R^T*f from
   the interface problem, where R are the Rigid Body Modes.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI1ComputeRhsE_Private(FETI ft)
{
  PetscErrorCode ierr;
  Subdomain      sd = ft->subdomain;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  ierr   = VecDestroy(&ft1->local_e);CHKERRQ(ierr);
  /* get number of rigid body modes */
  ierr   = MatMumpsGetInfog(ft->F_neumann,28,&ft->n_cs);CHKERRQ(ierr);
  if(ft->n_cs){
    /* compute matrix local_e */
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&ft1->local_e);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft1->rbm,sd->localRHS,ft1->local_e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1ComputeMatrixG_Private"
/*@
   FETI1ComputeMatrixG_Private - Computes the local matrix
   G=B*R, where R are the Rigid Body Modes.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI1ComputeMatrixG_Private(FETI ft)
{
  PetscErrorCode ierr;
  Subdomain      sd = ft->subdomain;
  Mat            x; 
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  ierr   = MatDestroy(&ft1->localG);CHKERRQ(ierr);
  /* get number of rigid body modes */
  ierr   = MatMumpsGetInfog(ft->F_neumann,28,&ft->n_cs);CHKERRQ(ierr);
  if(ft->n_cs){
    /* Compute rigid body modes */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,ft->n_cs,NULL,&ft1->rbm);CHKERRQ(ierr);
    ierr = MatDuplicate(ft1->rbm,MAT_DO_NOT_COPY_VALUES,&x);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(ft1->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ft1->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMumpsSetIcntl(ft->F_neumann,25,-1);CHKERRQ(ierr);
    ierr = MatMatSolve(ft->F_neumann,x,ft1->rbm);CHKERRQ(ierr);
    ierr = MatDestroy(&x);CHKERRQ(ierr);

    /* compute matrix localG */
    ierr = MatGetSubMatrix(ft1->rbm,sd->is_B_local,NULL,MAT_INITIAL_MATRIX,&x);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,ft->n_cs,NULL,&ft1->localG);CHKERRQ(ierr);
    ierr = MatMatMult(ft->B_delta,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&ft1->localG);CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1SetUpNeumannSolver_Private"
/*@
   FETI1SetUpNeumannSolver - It mainly configures the neumann direct solver and performes the factorization.

   Input Parameter:
.  feti - the FETI context

   Notes: 
   In a future this rutine could be moved to the FETI class.

   Level: developer

.keywords: FETI1

.seealso: FETISetUp_FETI1()
@*/
static PetscErrorCode FETI1SetUpNeumannSolver_Private(FETI ft)
{
  PetscErrorCode ierr;
  PC             pc;
  PetscBool      issbaij;
  Subdomain      sd = ft->subdomain;
  
  PetscFunctionBegin;
#if !defined(PETSC_HAVE_MUMPS)
    SETERRQ(PetscObjectComm((PetscObject)ft),1,"EINS only supports MUMPS for the solution of the Neumann problem");
#endif
  if (!ft->ksp_neumann) {
    ierr = KSPCreate(PETSC_COMM_SELF,&ft->ksp_neumann);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ft->ksp_neumann,(PetscObject)ft,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ft,(PetscObject)ft->ksp_neumann);CHKERRQ(ierr);
    ierr = KSPSetType(ft->ksp_neumann,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ft->ksp_neumann,&pc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)(sd->localA),MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
    if (issbaij) {
      ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    }
    ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
    ierr = KSPSetOperators(ft->ksp_neumann,sd->localA,sd->localA);CHKERRQ(ierr);
    /* prefix for setting options */
    ierr = KSPSetOptionsPrefix(ft->ksp_neumann,"feti1_neumann_");CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(sd->localA,"feti1_neumann_");CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&ft->F_neumann);CHKERRQ(ierr);
    /* sequential ordering */
    ierr = MatMumpsSetIcntl(ft->F_neumann,7,2);CHKERRQ(ierr);
    /* Null row pivot detection */
    ierr = MatMumpsSetIcntl(ft->F_neumann,24,1);CHKERRQ(ierr);
    /* threshhold for row pivot detection */
    ierr = MatMumpsSetCntl(ft->F_neumann,3,1.e-6);CHKERRQ(ierr);

    /* Maybe the following two options should be given as external options and not here*/
    ierr = KSPSetFromOptions(ft->ksp_neumann);CHKERRQ(ierr);
    ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(ft->ksp_neumann,sd->localA,sd->localA);CHKERRQ(ierr);
  }
  /* Set Up KSP for Neumann problem: here the factorization takes place!!! */
  ierr = KSPSetUp(ft->ksp_neumann);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1SetUpCoarseProblem_Private"
/*@
   FETI1SetUpCoarseProblem_Private - It mainly configures the coarse problem and factorizes it. It also creates alpha_local vector.

   Input Parameter:
.  feti - the FETI context

   Notes: 
   FETI1ComputeMatrixG_Private() should be called before calling FETI1SetUpCoarseProblem_Private().

   Notes regarding non-blocking communications in this rutine: 
   You must avoid reusing the send message buffer before the communication has been completed.

   Level: developer

.keywords: FETI1

.seealso: FETI1SetUpCoarseProblem_FETI1()
@*/
static PetscErrorCode FETI1SetUpCoarseProblem_Private(FETI ft)
{
  PetscErrorCode ierr;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  PetscMPIInt    i_mpi,i_mpi1,sizeG,size,*c_displ,*c_count,n_recv,n_send,rankG;
  PetscInt       k,k0,*c_coarse,*r_coarse,total_c_coarse,*idxm=NULL,*idxn=NULL;
  /* nnz: array containing the number of block nonzeros in the upper triangular plus diagonal portion of each block*/
  PetscInt       i,j,idx,*n_cs_comm,*nnz,size_floating,total_size_matrices=0,localnnz=0;
  PetscScalar    *m_pointer=NULL,*m_pointer1=NULL,**array=NULL;
  MPI_Comm       comm;
  MPI_Request    *send_reqs=NULL,*recv_reqs=NULL;
  IS             isindex;
  /* Gholder: for holding non-local G that I receive from neighbours*/
  /* submat: submatrices of my own G to send to my neighbours */
  /* result: result of the local multiplication G^T*G*/
  Mat            *submat,result,aux_mat;
  
  PetscFunctionBegin;  
  /* in the following rankG and sizeG are related to the MPI_Comm comm */
  /* whereas rank and size are related to the MPI_Comm floatingComm */
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rankG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&sizeG);CHKERRQ(ierr);
  
  /* computing n_cs_comm that is number of rbm per subdomain and the communicator of floating structures */
  ierr = PetscMalloc1(sizeG,&n_cs_comm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&ft->n_cs,1,MPIU_INT,n_cs_comm,1,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Comm_split(comm,(n_cs_comm[rankG]>0),rankG,&ft1->floatingComm);CHKERRQ(ierr);

  if (ft->n_cs){
    /* creates alpha_local vector for holding local coefficients for vector with rigid body modes */
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&ft1->alpha_local);CHKERRQ(ierr);
    /* compute size and rank to the new communicator */
    ierr = MPI_Comm_size(ft1->floatingComm,&size);CHKERRQ(ierr);
    /* computing displ_f and count_f_rbm */
    ierr                 = PetscMalloc1(size,&ft1->displ_f);CHKERRQ(ierr);
    ierr                 = PetscMalloc1(size,&ft1->count_f_rbm);CHKERRQ(ierr);
    ft1->displ_f[0]      = 0;
    ft1->count_f_rbm[0]  = n_cs_comm[0];
    k                    = (n_cs_comm[0]>0);
    for (i=1;i<sizeG;i++){
      if(n_cs_comm[i]) {
	ft1->count_f_rbm[k] = n_cs_comm[i];
	ft1->displ_f[k]     = ft1->displ_f[k-1] + ft1->count_f_rbm[k-1];
	k++;
      }
    }
  }
  
  /* computing displ structure for next allgatherv and total number of rbm of the system */
  ierr               = PetscMalloc1(sizeG,&ft1->displ);CHKERRQ(ierr);
  ierr               = PetscMalloc1(sizeG,&ft1->count_rbm);CHKERRQ(ierr);
  ft1->displ[0]      = 0;
  ft1->count_rbm[0]  = n_cs_comm[0];
  ft1->total_rbm     = n_cs_comm[0];
  size_floating      = (n_cs_comm[0]>0);
  for (i=1;i<sizeG;i++){
    ft1->total_rbm    += n_cs_comm[i];
    size_floating     += (n_cs_comm[i]>0);
    ft1->count_rbm[i]  = n_cs_comm[i];
    ft1->displ[i]      = ft1->displ[i-1] + ft1->count_rbm[i-1];
  }

  /* localnnz: nonzeros for my row of the coarse probem */
  localnnz            = ft->n_cs;
  total_size_matrices = 0;
  ft1->max_n_cs      = ft->n_cs;
  n_send              = (ft->n_neigh_lb-1)*(ft->n_cs>0);
  n_recv              = 0;
  if(ft->n_cs) {
    for (i=1;i<ft->n_neigh_lb;i++){
      i_mpi                = n_cs_comm[ft->neigh_lb[i]];
      n_recv              += (i_mpi>0);
      ft1->max_n_cs       = (ft1->max_n_cs > i_mpi) ? ft1->max_n_cs : i_mpi;
      total_size_matrices += i_mpi*ft->n_shared_lb[i];
      localnnz            += i_mpi*(ft->neigh_lb[i]>rankG);
    }
  } else {
    for (i=1;i<ft->n_neigh_lb;i++){
      i_mpi                = n_cs_comm[ft->neigh_lb[i]];
      n_recv              += (i_mpi>0);
      ft1->max_n_cs       = (ft1->max_n_cs > i_mpi) ? ft1->max_n_cs : i_mpi;
      total_size_matrices += i_mpi*ft->n_shared_lb[i];
    }
  }

  ierr = PetscMalloc1(ft1->total_rbm,&nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(size_floating,&idxm);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_displ);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_count);CHKERRQ(ierr);
  c_count[0]     = (n_cs_comm[0]>0);
  c_displ[0]     = 0;
  for (i=1;i<sizeG;i++) {
    c_displ[i] = c_displ[i-1] + c_count[i-1];
    c_count[i] = (n_cs_comm[i]>0);
  }
  ierr = MPI_Allgatherv(&localnnz,(n_cs_comm[rankG]>0),MPIU_INT,idxm,c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);
  for (k0=0,k=0,i=0;i<sizeG;i++) {
    for(j=0;j<ft1->count_rbm[i];j++) nnz[k++] = idxm[k0];
    k0 += (ft1->count_rbm[i]>0);
  }
  ierr = PetscFree(idxm);CHKERRQ(ierr);

  /* create the "global" matrix for holding G^T*G */
  if(ft1->destroy_coarse){ ierr = MatDestroy(&ft1->coarse_problem);CHKERRQ(ierr);}
  ierr = MatCreate(PETSC_COMM_SELF,&ft1->coarse_problem);CHKERRQ(ierr);
  ierr = MatSetType(ft1->coarse_problem,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(ft1->coarse_problem,1);CHKERRQ(ierr);
  ierr = MatSetSizes(ft1->coarse_problem,ft1->total_rbm,ft1->total_rbm,ft1->total_rbm,ft1->total_rbm);CHKERRQ(ierr);
  ierr = MatSetOption(ft1->coarse_problem,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(ft1->coarse_problem,1,PETSC_DEFAULT,nnz);CHKERRQ(ierr);
  ierr = MatSetUp(ft1->coarse_problem);CHKERRQ(ierr);

  /* Communicate matrices G */
  if(n_send) {
    ierr = PetscMalloc1(n_send,&send_reqs);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_send,&submat);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_send,&array);CHKERRQ(ierr);
    for (j=0, i=1; i<ft->n_neigh_lb; i++){
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(ft1->localG,isindex,NULL,MAT_INITIAL_MATRIX,&submat[j]);CHKERRQ(ierr);
      ierr = MatDenseGetArray(submat[j],&array[j]);CHKERRQ(ierr);   
      ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
      ierr = MPI_Isend(array[j],ft->n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&send_reqs[j]);CHKERRQ(ierr);
      ierr = ISDestroy(&isindex);CHKERRQ(ierr);
      j++;
    }
  }
  if(n_recv) {
    ierr = PetscMalloc1(total_size_matrices,&ft1->matrices);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_recv,&recv_reqs);CHKERRQ(ierr);
    for (j=0,idx=0,i=1; i<ft->n_neigh_lb; i++){
      if (n_cs_comm[ft->neigh_lb[i]]>0) {
	ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
	ierr = MPI_Irecv(&ft1->matrices[idx],n_cs_comm[ft->neigh_lb[i]]*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&recv_reqs[j]);CHKERRQ(ierr);    
	idx += n_cs_comm[ft->neigh_lb[i]]*ft->n_shared_lb[i]; 
	j++;
      }	  
    }
  }
  if(n_recv) { ierr = MPI_Waitall(n_recv,recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
  if(n_send) {
    ierr = MPI_Waitall(n_send,send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    for (i=0;i<n_send;i++) {
      ierr = MatDenseRestoreArray(submat[i],&array[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&submat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(array);CHKERRQ(ierr);
    ierr = PetscFree(submat);CHKERRQ(ierr);
  }

  if(n_recv) {
    /* store received matrices in Gholder */
    ft1->n_Gholder = n_recv;
    ierr = PetscMalloc1(n_recv,&ft1->Gholder);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_recv,&ft1->neigh_holder);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*n_recv,&ft1->neigh_holder[0]);CHKERRQ(ierr);
    for (i=1;i<n_recv;i++) { 
      ft1->neigh_holder[i] = ft1->neigh_holder[i-1] + 2;
    }
    for (i=0,idx=0,k=1; k<ft->n_neigh_lb; k++){
      if (n_cs_comm[ft->neigh_lb[k]]>0) {
	ft1->neigh_holder[i][0] = ft->neigh_lb[k];
	ft1->neigh_holder[i][1] = k;
	ierr  = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_shared_lb[k],n_cs_comm[ft->neigh_lb[k]],&ft1->matrices[idx],&ft1->Gholder[i++]);CHKERRQ(ierr);
	idx  += n_cs_comm[ft->neigh_lb[k]]*ft->n_shared_lb[k];
      }
    }
  }
  
  /** perfoming the actual multiplication G_{rankG}^T*G_{neigh_rankG>=rankG} */   
  if (ft->n_cs) {
    ierr = PetscMalloc1(ft->n_lambda_local,&idxm);CHKERRQ(ierr);
    ierr = PetscMalloc1(ft1->max_n_cs,&idxn);CHKERRQ(ierr);
    for (i=0;i<ft->n_lambda_local;i++) idxm[i] = i;
    for (i=0;i<ft->n_cs;i++) idxn[i] = i;
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,localnnz,NULL,&aux_mat);CHKERRQ(ierr);
    ierr = MatZeroEntries(aux_mat);CHKERRQ(ierr);
    ierr = MatSetOption(aux_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatDenseGetArray(ft1->localG,&m_pointer);CHKERRQ(ierr);
    ierr = MatSetValuesBlocked(aux_mat,ft->n_lambda_local,idxm,ft->n_cs,idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(ft1->localG,&m_pointer);CHKERRQ(ierr);

    for (k=0; k<ft1->n_Gholder; k++){
      j   = ft1->neigh_holder[k][0];
      idx = ft1->neigh_holder[k][1];
      if (j>rankG) {
	for (k0=0;k0<n_cs_comm[j];k0++) idxn[k0] = i++;
	ierr = MatDenseGetArray(ft1->Gholder[k],&m_pointer);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(aux_mat,ft->n_shared_lb[idx],ft->shared_lb[idx],n_cs_comm[j],idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(ft1->Gholder[k],&m_pointer);CHKERRQ(ierr);	
      }
    }
    ierr = MatAssemblyBegin(aux_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(aux_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_cs,localnnz,NULL,&result);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(ft1->localG,aux_mat,MAT_REUSE_MATRIX,PETSC_DEFAULT,&result);CHKERRQ(ierr);
    ierr = MatDestroy(&aux_mat);CHKERRQ(ierr);
    ierr = PetscFree(idxm);CHKERRQ(ierr);
    ierr = PetscFree(idxn);CHKERRQ(ierr);
    
    /* building structures for assembling the "global matrix" of the coarse problem */
    ierr   = PetscMalloc1(ft->n_cs,&idxm);CHKERRQ(ierr);
    ierr   = PetscMalloc1(localnnz,&idxn);CHKERRQ(ierr);
    /** row indices */
    idx = ft1->displ[rankG];
    for (i=0; i<ft->n_cs; i++) idxm[i] = i + idx;
    /** col indices */
    for (i=0; i<ft->n_cs; i++) idxn[i] = i + idx;
    for (j=1; j<ft->n_neigh_lb; j++) {
      k0  = n_cs_comm[ft->neigh_lb[j]];
      if ((ft->neigh_lb[j]>rankG)&&(k0>0)) {
	idx = ft1->displ[ft->neigh_lb[j]];
	for (k=0;k<k0;k++, i++) idxn[i] = k + idx;
      }
    }
    /** local "row block" contribution to G^T*G */
    ierr = MatDenseGetArray(result,&m_pointer);CHKERRQ(ierr);
  }
  
  /* assemble matrix */
  ierr           = PetscMalloc1(ft1->total_rbm,&r_coarse);CHKERRQ(ierr);
  total_c_coarse = nnz[0]*(n_cs_comm[0]>0);
  c_count[0]     = nnz[0]*(n_cs_comm[0]>0);
  c_displ[0]     = 0;
  idx            = n_cs_comm[0];
  for (i=1;i<sizeG;i++) {
    total_c_coarse += nnz[idx]*(n_cs_comm[i]>0);
    c_displ[i]      = c_displ[i-1] + c_count[i-1];
    c_count[i]      = nnz[idx]*(n_cs_comm[i]>0);
    idx            += n_cs_comm[i];
  }
  ierr = PetscMalloc1(total_c_coarse,&c_coarse);CHKERRQ(ierr);
  /* gather rows and columns*/
  ierr = MPI_Allgatherv(idxm,ft->n_cs,MPIU_INT,r_coarse,ft1->count_rbm,ft1->displ,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(idxn,localnnz,MPIU_INT,c_coarse,c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);

  /* gather values for the coarse problem's matrix and assemble it */
  ierr = MatZeroEntries(ft1->coarse_problem);CHKERRQ(ierr);
  for (k=0,k0=0,i=0; i<sizeG; i++) {
    if(n_cs_comm[i]>0){
      ierr = PetscMPIIntCast(n_cs_comm[i]*c_count[i],&i_mpi);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(i,&i_mpi1);CHKERRQ(ierr);
      if(i==rankG) {
	ierr = MPI_Bcast(m_pointer,i_mpi,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(ft1->coarse_problem,n_cs_comm[i],&r_coarse[k],c_count[i],&c_coarse[k0],m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(result,&m_pointer);CHKERRQ(ierr);
	ierr = MatDestroy(&result);CHKERRQ(ierr);
      } else {
	ierr = PetscMalloc1(i_mpi,&m_pointer1);CHKERRQ(ierr);	  
	ierr = MPI_Bcast(m_pointer1,i_mpi,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(ft1->coarse_problem,n_cs_comm[i],&r_coarse[k],c_count[i],&c_coarse[k0],m_pointer1,INSERT_VALUES);CHKERRQ(ierr);
	ierr = PetscFree(m_pointer1);CHKERRQ(ierr);
      }
      k0 += c_count[i];
      k  += n_cs_comm[i];
    }
  }
  ierr = MatAssemblyBegin(ft1->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ft1->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(idxm);CHKERRQ(ierr);
  ierr = PetscFree(idxn);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscFree(c_coarse);CHKERRQ(ierr);
  ierr = PetscFree(r_coarse);CHKERRQ(ierr);
  ierr = PetscFree(c_count);CHKERRQ(ierr);
  ierr = PetscFree(c_displ);CHKERRQ(ierr);
  ierr = PetscFree(send_reqs);CHKERRQ(ierr);
  ierr = PetscFree(recv_reqs);CHKERRQ(ierr);
  ierr = PetscFree(n_cs_comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1FactorizeCoarseProblem_Private"
/*@
  FETI1FactorizeCoarseProblem_Private - Factorizes the coarse problem. 

  Input Parameter:
  .  ft - the FETI context

   Notes: 
   FETI1SetUpCoarseProblem_Private() must be called before calling FETI1FactorizeCoarseProblem_Private().

   Level: developer

.keywords: FETI1

.seealso: FETI1SetUpCoarseProblem_Private()
@*/
static PetscErrorCode FETI1FactorizeCoarseProblem_Private(FETI ft)
{
  PetscErrorCode ierr;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  PC             pc;
  
  PetscFunctionBegin; 
  if(!ft1->coarse_problem) SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: FETI1SetUpCoarseProblem_Private() must be first called");

  /* factorize the coarse problem */
  ierr = KSPCreate(PETSC_COMM_SELF,&ft1->ksp_coarse);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ft1->ksp_coarse,(PetscObject)ft1,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ft1,(PetscObject)ft1->ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetType(ft1->ksp_coarse,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ft1->ksp_coarse,"feti1_pc_coarse_");CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ft1->coarse_problem,"feti1_pc_coarse_");CHKERRQ(ierr);
  ierr = KSPGetPC(ft1->ksp_coarse,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ft1->ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOperators(ft1->ksp_coarse,ft1->coarse_problem,ft1->coarse_problem);CHKERRQ(ierr);
  ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
  ierr = KSPSetUp(ft1->ksp_coarse);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&ft1->F_coarse);CHKERRQ(ierr);
  if(ft1->destroy_coarse) {
    ierr = MatDestroy(&ft1->coarse_problem);CHKERRQ(ierr);
    ft1->coarse_problem = 0;    
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1ApplyCoarseProblem_Private"
/*@
   FETI1ApplyCoarseProblem_Private - Applies the operation G*(G^T*G)^{-1} to a vector  

   Input Parameter:
.  ft       - the FETI context
.  v  - the input vector (in this case v is a local copy of the global assembled vector)

   Output Parameter:
.  r  - the output vector. 

   Level: developer

.keywords: FETI1

.seealso: FETI1ApplyCoarseProblem_Private()
@*/
static PetscErrorCode FETI1ApplyCoarseProblem_Private(FETI ft,Vec v,Vec r)
{
  PetscErrorCode     ierr;
  FETI_1             *ft1 = (FETI_1*)ft->data;
  Vec                v_rbm; /* vec of dimension total_rbm */
  Vec                v0;    /* vec of dimension n_cs */
  Vec                r_local,vec_holder;
  IS                 subset;
  PetscMPIInt        rank;
  MPI_Comm           comm;
  PetscInt           i,j,idx0,idx1,*indices;
  const PetscScalar  *m_pointer; 
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  if(!ft1->F_coarse) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: FETI1FactorizeCoarseProblem_Private() must be first called");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* apply (G^T*G)^{-1}: compute v_rbm = (G^T*G)^{-1}*v */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->total_rbm,&v_rbm);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(ft1->F_coarse,25,0);CHKERRQ(ierr);
#endif
  ierr = MatSolve(ft1->F_coarse,v,v_rbm);CHKERRQ(ierr);

  /* apply G: compute r = G*v_rbm */
  ierr = VecUnAsmGetLocalVector(r,&r_local);CHKERRQ(ierr);
  ierr = PetscMalloc1(ft1->max_n_cs,&indices);CHKERRQ(ierr);
  /** mulplying by localG for the current processor */
  if(ft->n_cs) {
    for (i=0;i<ft->n_cs;i++) indices[i] = ft1->displ[rank] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_cs,indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
    ierr = VecGetSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = MatMult(ft1->localG,v0,r_local);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = ISDestroy(&subset);CHKERRQ(ierr);
  } else {
    ierr = VecSet(r_local,0.0);CHKERRQ(ierr);
  }
  /** multiplying by localG for the other processors */
  for (j=0;j<ft1->n_Gholder;j++) {
    idx0 = ft1->neigh_holder[j][0];
    idx1 = ft1->neigh_holder[j][1];   
    for (i=0;i<ft1->count_rbm[idx0];i++) indices[i] = ft1->displ[idx0] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft1->count_rbm[idx0],indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
    ierr = VecGetSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_shared_lb[idx1],&vec_holder);CHKERRQ(ierr);
    ierr = MatMult(ft1->Gholder[j],v0,vec_holder);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec_holder,&m_pointer);CHKERRQ(ierr);
    ierr = VecSetValues(r_local,ft->n_shared_lb[idx1],ft->shared_lb[idx1],m_pointer,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = ISDestroy(&subset);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec_holder,&m_pointer);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_holder);CHKERRQ(ierr); 
  }
  
  ierr = VecAssemblyBegin(r_local);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(r_local);CHKERRQ(ierr);

  ierr = VecUnAsmRestoreLocalVector(r,r_local);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = VecDestroy(&v_rbm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1ComputeInitialCondition_Private"
/*@
   FETI1ComputeInitialCondition_Private - Computes initial condition
   for the interface problem. Once the initial condition is computed
   local_e is destroyed.

   Input Parameter:
.  ft - the FETI context

   Level: developer

.keywords: FETI1

@*/
static PetscErrorCode FETI1ComputeInitialCondition_Private(FETI ft)
{
  PetscErrorCode    ierr;
  FETI_1            *ft1 = (FETI_1*)ft->data;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) { ierr = VecGetArrayRead(ft1->local_e,&sbuff);CHKERRQ(ierr);}
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,ft1->count_rbm,ft1->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft->n_cs) { ierr = VecRestoreArrayRead(ft1->local_e,&sbuff);CHKERRQ(ierr);}
  ierr = FETI1ApplyCoarseProblem_Private(ft,asm_e,ft->lambda_global);CHKERRQ(ierr);
  if (ft->n_cs) { ierr = VecDestroy(&ft1->local_e);CHKERRQ(ierr);}
  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1Project_RBM"
/*@
   FETI1Project_RBM - Performs the projection step of FETI1.

   Input Parameter:
.  ft        - the FETI context
.  g_global  - the vector to project
.  y         - the projected vector

   Level: developer

.keywords: FETI1

@*/
PetscErrorCode FETI1Project_RBM(void* ft_ctx, Vec g_global, Vec y)
{
  PetscErrorCode    ierr;
  FETI              ft; 
  FETI_1            *ft1;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  Vec               lambda_local,localv;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft_ctx,FETI_CLASSID,1);
  PetscValidHeaderSpecific(g_global,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ft   = (FETI)ft_ctx;
  ft1  = (FETI_1*)ft->data;
  comm = PetscObjectComm((PetscObject)ft);
  if (y == g_global) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use g_global == y");
  ierr = VecUnAsmGetLocalVectorRead(g_global,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft1->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecUnAsmRestoreLocalVectorRead(g_global,lambda_local);CHKERRQ(ierr);
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,ft1->count_rbm,ft1->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETI1ApplyCoarseProblem_Private(ft,asm_e,y);CHKERRQ(ierr);
  ierr = VecAYPX(y,-1,g_global);CHKERRQ(ierr);

  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISolve_FETI1"
/*@
   FETISolve_FETI1 - Computes the primal solution using the FETI1 method.

   Input: 
.  ft - the FETI context

   Output:
.  u  - vector to store the solution

   Level: beginner

.keywords: FETI1
@*/
static PetscErrorCode FETISolve_FETI1(FETI ft, Vec u){
  PetscErrorCode    ierr;
  FETI_1            *ft1 = (FETI_1*)ft->data;
  Subdomain         sd = ft->subdomain;
  Vec               lambda_local;
  
  PetscFunctionBegin;
  /* Solve interface problem */
  ierr = KSPSolve(ft->ksp_interface,ft->d,ft->lambda_global);CHKERRQ(ierr);
  /* Get residual of the interface problem */
  ierr = KSPGetResidual(ft->ksp_interface,&ft1->res_interface);CHKERRQ(ierr);
  /* compute alpha_local */
  ierr = FETI1ComputeAlpha_Private(ft);CHKERRQ(ierr);
  /* computing B_delta^T*lambda */
  ierr = VecUnAsmGetLocalVectorRead(ft->lambda_global,&lambda_local);CHKERRQ(ierr);
  ierr = MatMultTranspose(ft->B_delta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(ft->lambda_global,lambda_local);CHKERRQ(ierr);
  /* computing f - B_delta^T*lambda */
  ierr = VecAYPX(sd->vec1_N,-1.0,sd->localRHS);CHKERRQ(ierr);   
  /* Application of the already factorized pseudo-inverse */
  ierr = MatMumpsSetIcntl(ft->F_neumann,25,0);CHKERRQ(ierr);
  ierr = MatSolve(ft->F_neumann,sd->vec1_N,u);CHKERRQ(ierr);
  if (ft->n_cs) {
    /* computing R*alpha */
    ierr = MatMult(ft1->rbm,ft1->alpha_local,sd->vec1_N);CHKERRQ(ierr);
    /* computing u = A^+*(f - B_delta^T*lambda) + R*alpha */
    ierr = VecAXPY(u,-1.0,sd->vec1_N);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1ComputeAlpha_Private"
/*@
   FETI1ComputeAlpha_Private - Computes the local coefficients multiplying the correspoding local vector with Rigid Body Modes.

   Input: 
.  ft - the FETI context

   Level: beginner

.keywords: FETI1
@*/
static PetscErrorCode FETI1ComputeAlpha_Private(FETI ft)
{
  PetscErrorCode    ierr;
  FETI_1            *ft1 = (FETI_1*)ft->data;
  Vec               alpha_g,asm_g; 
  PetscMPIInt       rank;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  Vec               lambda_local;
  
  PetscFunctionBegin;
  ierr = VecUnAsmGetLocalVectorRead(ft1->res_interface,&lambda_local);CHKERRQ(ierr);
  if (!ft->n_cs)  PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(ft1->floatingComm,&rank);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->total_rbm,&alpha_g);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->total_rbm,&asm_g);CHKERRQ(ierr);  
  ierr = MatMultTranspose(ft1->localG,lambda_local,ft1->alpha_local);CHKERRQ(ierr);   
  ierr = VecGetArrayRead(ft1->alpha_local,&sbuff);CHKERRQ(ierr);
  ierr = VecGetArray(asm_g,&rbuff);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,ft1->count_f_rbm,ft1->displ_f,MPIU_SCALAR,ft1->floatingComm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_g,&rbuff);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ft1->alpha_local,&sbuff);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(ft1->res_interface,lambda_local);CHKERRQ(ierr);
  
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(ft1->F_coarse,25,0);CHKERRQ(ierr);
#endif
  ierr = MatSolve(ft1->F_coarse,asm_g,alpha_g);CHKERRQ(ierr);

  ierr = VecGetArrayRead(alpha_g,&sbuff);CHKERRQ(ierr);
  ierr = VecGetArray(ft1->alpha_local,&rbuff);CHKERRQ(ierr);
  ierr = PetscMemcpy(rbuff,sbuff+ft1->displ_f[rank],sizeof(PetscScalar)*ft1->count_f_rbm[rank]);CHKERRQ(ierr);
  ierr = VecRestoreArray(ft1->alpha_local,&rbuff);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(alpha_g,&sbuff);CHKERRQ(ierr);
  ierr = VecDestroy(&asm_g);CHKERRQ(ierr);
  ierr = VecDestroy(&alpha_g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1SetDefaultOptions"
/*@
   FETI1SetDefaultOptions - Sets default options for the FETI1
   solver. Mainly, it sets every KSP to MUMPS and sets fully redudant
   lagrange multipliers.

   Input: Input taken by PetscOptionsInsert()
.  argc   -  number of command line arguments
.  args   -  the command line arguments
.  file   -  optional file

   Level: beginner

.keywords: FETI1

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode FETI1SetDefaultOptions(int *argc,char ***args,const char file[])
{
  PetscErrorCode    ierr;
  char mumps_options[]        = "-feti_pc_dirichlet_pc_factor_mat_solver_package mumps \
                                 -feti_pc_dirichlet_mat_mumps_icntl_7 2                \
                                 -feti1_pc_coarse_pc_factor_mat_solver_package mumps   \
                                 -feti1_pc_coarse_mat_mumps_icntl_7 2";
  char other_options[]        = "-feti_fullyredundant             \
                                 -feti_scaling_type scmultiplicity \
                                 -feti1_destroy_coarse";
  
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscOptionsInsertString(NULL,mumps_options);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsInsertString(NULL,other_options);CHKERRQ(ierr);
  ierr = PetscOptionsInsert(NULL,argc,args,file);CHKERRQ(ierr);
    
  PetscFunctionReturn(0);
}
