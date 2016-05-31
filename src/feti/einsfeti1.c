#include <../src/feti/einsfeti1.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einssys.h>


/* private functions*/
static PetscErrorCode FETI1SetUpNeumannSolver_Private(FETI);
static PetscErrorCode FETI1BuildInterfaceProblem_Private(FETI);
static PetscErrorCode FETIDestroy_FETI1(FETI);
static PetscErrorCode FETISetUp_FETI1(FETI);
static PetscErrorCode FETI1DestroyMatF_Private(Mat);
static PetscErrorCode FETI1MatMult_Private(Mat,Vec,Vec);
static PetscErrorCode FETISolve_FETI1(FETI,Vec);
static PetscErrorCode FETI1SetInterfaceProblemRHS_Private(FETI);
  

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
  
  PetscFunctionBegin;
  if (!ft1) PetscFunctionReturn(0);
  ierr = VecDestroy(&ft1->local_e);CHKERRQ(ierr);
  ierr = VecDestroy(&ft1->alpha_local);CHKERRQ(ierr);
  ierr = MatDestroy(&ft1->rbm);CHKERRQ(ierr);
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
  FETI_1            *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  if (ft->state==FETI_STATE_INITIAL) {
    ierr = FETIScalingSetUp(ft);CHKERRQ(ierr);
    ierr = FETIBuildLambdaAndB(ft);CHKERRQ(ierr);
    ierr = FETI1SetUpNeumannSolver_Private(ft);CHKERRQ(ierr);
    ierr = FETI1BuildInterfaceProblem_Private(ft);CHKERRQ(ierr);
    ierr = FETIBuildInterfaceKSP(ft);CHKERRQ(ierr);

    ierr = FETICSSetUp(ft->ftcs);CHKERRQ(ierr);
    ierr = FETICSComputeCoarseBasis(ft->ftcs,&ft->localG,&ft1->rbm);CHKERRQ(ierr);

    /* compute matrix local_e */
    if(ft->n_cs){
      ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&ft1->local_e);CHKERRQ(ierr);
      ierr = MatMultTranspose(ft1->rbm,sd->localRHS,ft1->local_e);CHKERRQ(ierr);
    }
    ierr = FETI1SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);

    /* setup projection */
    ierr = FETIPJSetUp(ft->ftpj);CHKERRQ(ierr);
    ierr = FETIPJFactorizeCoarseProblem(ft->ftpj);CHKERRQ(ierr);
    /* creates alpha_local vector for holding local coefficients for vector with rigid body modes */
    if (ft->n_cs){ ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&ft1->alpha_local);CHKERRQ(ierr); }
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
    /* compute matrix local_e */
    if(ft->n_cs){ ierr = MatMultTranspose(ft1->rbm,sd->localRHS,ft1->local_e);CHKERRQ(ierr);}
    ierr = FETI1SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);
  }

  ierr = FETIPJComputeInitialCondition(ft->ftpj,ft1->local_e,ft->lambda_global);CHKERRQ(ierr);  
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
.  -feti_pj1level_destroy_coarse - If set, the matrix of the coarse problem, that is (G^T*G) or (G^T*Q*G), will be destroyed after factorization.
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
  ft->ftcs_type                = CS_RIGID_BODY_MODES;
  /* function pointers */
  ft->ops->setup               = FETISetUp_FETI1;
  ft->ops->destroy             = FETIDestroy_FETI1;
  ft->ops->setfromoptions      = 0;
  ft->ops->computesolution     = FETISolve_FETI1;
  ft->ops->view                = 0;

  ft->ftpj_type = PJ_FIRST_LEVEL;
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
  if(ft->n_cs) { ierr = FETIPJComputeAlphaNullSpace(ft->ftpj,ft1->res_interface,ft1->alpha_local);CHKERRQ(ierr);}
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
                                 -feti_pj1level_pc_coarse_pc_factor_mat_solver_package mumps   \
                                 -feti_pj1level_pc_coarse_mat_mumps_icntl_7 2";
  char other_options[]        = "-feti_fullyredundant             \
                                 -feti_scaling_type scmultiplicity \
                                 -feti_pj1level_destroy_coarse";
  
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscOptionsInsertString(NULL,mumps_options);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsInsertString(NULL,other_options);CHKERRQ(ierr);
  ierr = PetscOptionsInsert(NULL,argc,args,file);CHKERRQ(ierr);
    
  PetscFunctionReturn(0);
}
