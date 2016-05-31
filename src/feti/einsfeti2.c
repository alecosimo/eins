#include <../src/feti/einsfeti2.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einspc.h>
#include <einssys.h>


/* private functions*/
static PetscErrorCode FETI2SetUpNeumannSolver_Private(FETI);
static PetscErrorCode FETI2DestroyMatF_Private(Mat);
static PetscErrorCode FETI2MatMult_Private(Mat,Vec,Vec);
static PetscErrorCode FETI2BuildInterfaceProblem_Private(FETI);
static PetscErrorCode FETIDestroy_FETI2(FETI);
static PetscErrorCode FETISetUp_FETI2(FETI);
static PetscErrorCode FETISolve_FETI2(FETI,Vec);
static PetscErrorCode FETI2SetInterfaceProblemRHS_Private(FETI);


#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI2"
static PetscErrorCode FETIDestroy_FETI2(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  
  PetscFunctionBegin;
  if (!ft2) PetscFunctionReturn(0);

  ierr = PetscFree(ft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETI2"
static PetscErrorCode FETISetUp_FETI2(FETI ft)
{
  PetscErrorCode    ierr;
  Subdomain         sd = ft->subdomain;
  PetscObjectState  mat_state;
  PetscBool         flg;
  
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)ft->ftcs,CS_NONE,&flg);CHKERRQ(ierr);
  if (ft->state==FETI_STATE_INITIAL) {
    ierr = FETIScalingSetUp(ft);CHKERRQ(ierr);
    ierr = FETIBuildLambdaAndB(ft);CHKERRQ(ierr);
    ierr = FETI2SetUpNeumannSolver_Private(ft);CHKERRQ(ierr);
    ierr = FETI2BuildInterfaceProblem_Private(ft);CHKERRQ(ierr);
    ierr = FETIBuildInterfaceKSP(ft);CHKERRQ(ierr); /* the PC for the interface problem is setup here */
    if (PetscNot(flg)) {
      ierr = FETICSSetUp(ft->ftcs);CHKERRQ(ierr);
      ierr = FETICSComputeCoarseBasis(ft->ftcs,&ft->localG,NULL);CHKERRQ(ierr);
    }
    ierr = FETI2SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);

    /* set projection in ksp */
    //    if (PetscNot(flg)) {
      ierr = FETIPJSetUp(ft->ftpj);CHKERRQ(ierr);     
      ierr = FETIPJGatherNeighborsCoarseBasis(ft->ftpj);CHKERRQ(ierr);
      ierr = FETIPJAssembleCoarseProblem(ft->ftpj);CHKERRQ(ierr);
      ierr = FETIPJFactorizeCoarseProblem(ft->ftpj);CHKERRQ(ierr);
      //    }
  } else {
    ierr = PetscObjectStateGet((PetscObject)sd->localA,&mat_state);CHKERRQ(ierr);
    if (mat_state>ft->mat_state) {
      ierr = PetscObjectStateSet((PetscObject)ft->F,mat_state);CHKERRQ(ierr);  
      if (ft->resetup_pc_interface) {
	PC pc;
	ierr = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
	ierr = PCSetUp(pc);CHKERRQ(ierr);
      }
      ierr = FETI2SetUpNeumannSolver_Private(ft);CHKERRQ(ierr);
      if (ft->resetup_pc_interface && PetscNot(flg)) {
	ierr = FETICSComputeCoarseBasis(ft->ftcs,&ft->localG,NULL);CHKERRQ(ierr);
	ierr = FETIPJGatherNeighborsCoarseBasis(ft->ftpj);CHKERRQ(ierr);
      }
      //if (PetscNot(flg)) {
      ierr = FETIPJAssembleCoarseProblem(ft->ftpj);CHKERRQ(ierr);
	ierr = FETIPJFactorizeCoarseProblem(ft->ftpj);CHKERRQ(ierr);
	//}
      ft->mat_state = mat_state;
    }
    ierr = FETI2SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);
  }

  ierr = FETIPJComputeInitialCondition(ft->ftpj);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "FETICreate_FETI2"
/*@
   FETI2 - Implementation of the FETI-1 method. Some comments about options can be put here!

   Options database:
.  -feti_fullyredundant: use fully redundant Lagrange multipliers.
.  -feti_interface_<ksp or pc option>: options for the KSP for the interface problem
.  -feti2_neumann_<ksp or pc option>: for setting pc and ksp options for the neumann solver. 
.  -feti_pc_dirichilet_<ksp or pc option>: options for the KSP or PC to use for solving the Dirichlet problem
   associated to the Dirichlet preconditioner
.  -feti_scaling_type - Sets the scaling type
.  -feti_scaling_factor - Sets a scaling factor different from one
.  -feti2_pc_coarse_<ksp or pc option>: options for the KSP for the coarse problem
.  -feti2_geneo_<option>: options for FETI2 using GENEO modes (e.g.:-feti2_geneo_eps_nev sets the number of eigenvalues)

   Level: beginner

.keywords: FETI, FETI-2
@*/
PetscErrorCode FETICreate_FETI2(FETI ft);
PetscErrorCode FETICreate_FETI2(FETI ft)
{
  PetscErrorCode      ierr;
  FETI_2*             feti2;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ft,&feti2);CHKERRQ(ierr);
  ft->data  = (void*)feti2;
  ierr      = PetscMemzero(feti2,sizeof(FETI_2));CHKERRQ(ierr);
  
  /* function pointers */
  ft->ops->setup               = FETISetUp_FETI2;
  ft->ops->destroy             = FETIDestroy_FETI2;
  ft->ops->setfromoptions      = 0;
  ft->ops->computesolution     = FETISolve_FETI2;
  ft->ops->view                = 0;

  ft->ftpj_type = PJ_SECOND_LEVEL;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "FETI2DestroyMatF_Private"
/*@
  FETI2DestroyMatF_Private - Destroy function for the MatShell matrix defining the interface problem's matrix F

   Input Parameters:
.  A - the Matrix context

   Level: developer

.seealso FETI2BuildInterfaceProblem_Private
@*/
static PetscErrorCode FETI2DestroyMatF_Private(Mat A)
{
  FETIMat_ctx    mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellUnAsmGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2MatMult_Private"
/*@
  FETI2MatMult_Private - MatMult function for the MatShell matrix defining the interface problem's matrix F. 
  It performes the product y=F*lambda_global

   Input Parameters:
.  F             - the Matrix context
.  lambda_global - vector to be multiplied by the matrix
.  y             - vector where to save the result of the multiplication

   Level: developer

.seealso FETI2BuildInterfaceProblem_Private
@*/
static PetscErrorCode FETI2MatMult_Private(Mat F, Vec lambda_global, Vec y) /* y=F*lambda_global */
{
  FETIMat_ctx    mat_ctx;
  FETI           ft; 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellUnAsmGetContext(F,(void**)&mat_ctx);CHKERRQ(ierr);
  ft   = mat_ctx->ft;
  ierr = MatMultFlambda_FETI(ft,lambda_global,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2MatGetVecs_Private"
static PetscErrorCode FETI2MatGetVecs_Private(Mat mat,Vec *right,Vec *left)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  FETI           ft  = NULL;
  FETIMat_ctx    mat_ctx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if(PetscNot(flg)) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot create vectors from non-shell matrix");
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
#define __FUNCT__ "FETI2SetInterfaceProblemRHS_Private"
/*@
   FETI2SetInterfaceProblemRHS_Private - Sets the RHS vector (vector d) of the interface problem.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI2SetInterfaceProblemRHS_Private(FETI ft)
{
  PetscErrorCode ierr;
  Subdomain      sd = ft->subdomain;
  Vec            d_local;
  
  PetscFunctionBegin;
  /** Application of the already factorized pseudo-inverse */
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
#define __FUNCT__ "FETI2BuildInterfaceProblem_Private"
/*@
   FETI2BuildInterfaceProblem_Private - Builds the interface problem, that is the matrix F and the vector d.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI2BuildInterfaceProblem_Private(FETI ft)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  /* Create the MatShell for F */
  ierr = FETICreateFMat(ft,(void (*)(void))FETI2MatMult_Private,(void (*)(void))FETI2DestroyMatF_Private,(void (*)(void))FETI2MatGetVecs_Private);CHKERRQ(ierr);
  /* Creating vector d for the interface problem */
  ierr = MatCreateVecs(ft->F,NULL,&ft->d);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2SetUpNeumannSolver_Private"
/*@
   FETI2SetUpNeumannSolver - It mainly configures the neumann direct solver and performes the factorization.

   Input Parameter:
.  feti - the FETI context

   Notes: 
   In a future this rutine could be moved to the FETI class.

   Level: developer

.keywords: FETI2

.seealso: FETISetUp_FETI2()
@*/
static PetscErrorCode FETI2SetUpNeumannSolver_Private(FETI ft)
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
    ierr = KSPSetOptionsPrefix(ft->ksp_neumann,"feti2_neumann_");CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(sd->localA,"feti2_neumann_");CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&ft->F_neumann);CHKERRQ(ierr);
    /* sequential ordering */
    ierr = MatMumpsSetIcntl(ft->F_neumann,7,2);CHKERRQ(ierr);
    /* Maybe the following two options should be given as external options and not here*/
    ierr = KSPSetFromOptions(ft->ksp_neumann);CHKERRQ(ierr);
    ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(ft->ksp_neumann,sd->localA,sd->localA);CHKERRQ(ierr);
  }
  ierr = KSPSetUp(ft->ksp_neumann);CHKERRQ(ierr);
    
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISolve_FETI2"
/*@
   FETISolve_FETI2 - Computes the primal solution using the FETI2 method.

   Input: 
.  ft - the FETI context

   Output:
.  u  - vector to store the solution

   Level: beginner

.keywords: FETI2
@*/
static PetscErrorCode FETISolve_FETI2(FETI ft, Vec u){
  PetscErrorCode    ierr;
  Subdomain         sd = ft->subdomain;
  Vec               lambda_local;
  
  PetscFunctionBegin;
  /* Solve interface problem */
  ierr = KSPSolve(ft->ksp_interface,ft->d,ft->lambda_global);CHKERRQ(ierr);
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
  ierr = MatSolve(ft->F_neumann,sd->vec1_N,u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2SetDefaultOptions"
/*@
   FETI2SetDefaultOptions - Sets default options for the FETI2
   solver. Mainly, it sets every KSP to MUMPS and sets fully redudant
   lagrange multipliers.

   Input: Input taken by PetscOptionsInsert()
.  argc   -  number of command line arguments
.  args   -  the command line arguments
.  file   -  optional file

   Level: beginner

.keywords: FETI2

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode FETI2SetDefaultOptions(int *argc,char ***args,const char file[])
{
  PetscErrorCode    ierr;
  char mumps_options[]        = "-feti_pc_dirichlet_pc_factor_mat_solver_package mumps \
                                 -feti_pc_dirichlet_mat_mumps_icntl_7 2                \
                                 -feti2_neumann_pc_factor_mat_solver_package mumps     \
                                 -feti2_neumann_mat_mumps_icntl_7 2                    \
                                 -feti_pj2level_pc_coarse_pc_factor_mat_solver_package mumps   \
                                 -feti_pj2level_pc_coarse_mat_mumps_icntl_7 2";
  char other_options[]        = "-feti_fullyredundant             \
                                 -feti_scaling_type scmultiplicity";
  
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscOptionsInsertString(NULL,mumps_options);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsInsertString(NULL,other_options);CHKERRQ(ierr);
  ierr = PetscOptionsInsert(NULL,argc,args,file);CHKERRQ(ierr);
    
  PetscFunctionReturn(0);
}

