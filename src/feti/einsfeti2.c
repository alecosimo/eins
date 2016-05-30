#include <../src/feti/einsfeti2.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einspc.h>
#include <einssys.h>


/* private functions*/
static PetscErrorCode FETI2SetUpNeumannSolver_Private(FETI);
static PetscErrorCode FETI2BuildInterfaceProblem_Private(FETI);
static PetscErrorCode FETIDestroy_FETI2(FETI);
static PetscErrorCode FETISetUp_FETI2(FETI);
static PetscErrorCode FETI2DestroyMatF_Private(Mat);
static PetscErrorCode FETI2MatMult_Private(Mat,Vec,Vec);
static PetscErrorCode FETI2SetUpCoarseProblem_RBM(FETI);
static PetscErrorCode FETI2ComputeCoarseProblem_RBM(FETI);
static PetscErrorCode FETI2FactorizeCoarseProblem_Private(FETI);
static PetscErrorCode FETI2ApplyCoarseProblem_Private(FETI,Vec,Vec);
static PetscErrorCode FETI2ComputeInitialCondition_RBM(FETI);
static PetscErrorCode FETI2ComputeInitialCondition_NOCOARSE(FETI);
static PetscErrorCode FETISolve_FETI2(FETI,Vec);
static PetscErrorCode FETI2SetInterfaceProblemRHS_Private(FETI);
static PetscErrorCode FETIDestroy_FETI2_RBM(FETI);
static PetscErrorCode MultFv_Private(FETI,Vec,Vec);
static PetscErrorCode FETIDestroy_FETI2_GATHER_NEIGH(FETI);
static PetscErrorCode FETI2GatherNeighborsG_Private(FETI);

PetscErrorCode FETI2Project_RBM(void*,Vec,Vec);
PetscErrorCode FETI2ReProject_RBM(void*,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI2"
/*@
   FETIDestroy_FETI2 - Destroys the FETI-1 context

   Input Parameters:
.  ft - the FETI context

.seealso FETICreate_FETI2
@*/
static PetscErrorCode FETIDestroy_FETI2(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  PetscInt       i;
  PetscBool      flg;
  
  PetscFunctionBegin;
  if (!ft2) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)ft->ftcs,CS_NONE,&flg);CHKERRQ(ierr);
  if (PetscNot(flg)) {
    ierr = FETIDestroy_FETI2_GATHER_NEIGH(ft);CHKERRQ(ierr);
    ierr = FETIDestroy_FETI2_RBM(ft);CHKERRQ(ierr);
  }

  ierr = FETICSDestroy(&ft->ftcs);CHKERRQ(ierr);
  
  ierr = MatDestroy(&ft2->localG);CHKERRQ(ierr);
  ierr = KSPDestroy(&ft2->ksp_coarse);CHKERRQ(ierr);
  if(ft2->neigh_holder) {
    ierr = PetscFree(ft2->neigh_holder[0]);CHKERRQ(ierr);
    ierr = PetscFree(ft2->neigh_holder);CHKERRQ(ierr);
  }
  if(ft2->displ) { ierr = PetscFree(ft2->displ);CHKERRQ(ierr);}
  for (i=0;i<ft2->n_Gholder;i++) {
    ierr = MatDestroy(&ft2->Gholder[i]);CHKERRQ(ierr);
  }
  if(ft2->coarse_problem) {ierr = MatDestroy(&ft2->coarse_problem);CHKERRQ(ierr);}
  ierr = PetscFree(ft2->Gholder);CHKERRQ(ierr);
  if(ft2->matrices) { ierr = PetscFree(ft2->matrices);CHKERRQ(ierr);}
  ierr = PetscFree(ft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETI2"
/*@
   FETISetUp_FETI2 - Prepares the structures needed by the FETI-1 solver.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETISetUp_FETI2(FETI ft)
{
  PetscErrorCode    ierr;
  FETI_2            *ft2 = (FETI_2*)ft->data;
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
      ierr = FETICSComputeCoarseBasisI(ft->ftcs,&ft2->localG);CHKERRQ(ierr);
    }
    
    ierr = FETI2SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);
    /* set projection in ksp */
    if (PetscNot(flg)) {
      ierr = KSPSetProjection(ft->ksp_interface,FETI2Project_RBM,(void*)ft);CHKERRQ(ierr);
      ierr = KSPSetReProjection(ft->ksp_interface,FETI2ReProject_RBM,(void*)ft);CHKERRQ(ierr);
      ierr = FETI2SetUpCoarseProblem_RBM(ft);CHKERRQ(ierr);
      ierr = FETI2GatherNeighborsG_Private(ft);CHKERRQ(ierr);
      ierr = FETI2ComputeCoarseProblem_RBM(ft);CHKERRQ(ierr);
      ierr = FETI2FactorizeCoarseProblem_Private(ft);CHKERRQ(ierr);
    }
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
	ierr = FETICSComputeCoarseBasisI(ft->ftcs,&ft2->localG);CHKERRQ(ierr);
	ierr = FETI2GatherNeighborsG_Private(ft);CHKERRQ(ierr);
      }
      if (PetscNot(flg)) {
	ierr = FETI2ComputeCoarseProblem_RBM(ft);CHKERRQ(ierr);
	ierr = FETI2FactorizeCoarseProblem_Private(ft);CHKERRQ(ierr);
      }
      ft->mat_state = mat_state;
    }
    ierr = FETI2SetInterfaceProblemRHS_Private(ft);CHKERRQ(ierr);
  }
  
  if (PetscNot(flg)) {
    ierr = FETI2ComputeInitialCondition_RBM(ft);CHKERRQ(ierr);
  } else {
    ierr = FETI2ComputeInitialCondition_NOCOARSE(ft);CHKERRQ(ierr);
  }
  
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
  ierr = MultFv_Private(ft,lambda_global,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MultFv_Private"
static PetscErrorCode MultFv_Private(FETI ft, Vec lambda_global, Vec y) /* y=F*lambda_global */
{
  Subdomain      sd;
  Vec            lambda_local,y_local;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sd   = ft->subdomain;
  ierr = VecUnAsmGetLocalVectorRead(lambda_global,&lambda_local);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(y,&y_local);CHKERRQ(ierr);
  /* Application of B_delta^T */
  ierr = MatMultTranspose(ft->B_delta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* Application of the already factorized pseudo-inverse */
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
#define __FUNCT__ "FETI2GatherNeighborsG_Private"
/*@
   FETI2GatherNeighborsG_Private - Sets up structures for gathering G
   matrices from neighbors.

   Input Parameter:
.  feti - the FETI context

   Level: developer

.keywords: FETI2
@*/
static PetscErrorCode FETI2GatherNeighborsG_Private(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  PetscMPIInt    i_mpi;
  PetscInt       i,j,n_cs,idx;
  PetscScalar    **array=NULL;
  MPI_Comm       comm;
  IS             isindex;
  Mat            *submat=NULL;
  
  PetscFunctionBegin;  
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  /* Communicate matrices G */
  if(ft2->n_send) {
    ierr = PetscMalloc1(ft2->n_send,&submat);CHKERRQ(ierr);
    ierr = PetscMalloc1(ft2->n_send,&array);CHKERRQ(ierr);
    for (j=0, i=1; i<ft->n_neigh_lb; i++){
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(ft2->localG,isindex,NULL,MAT_INITIAL_MATRIX,&submat[j]);CHKERRQ(ierr);
      ierr = MatDenseGetArray(submat[j],&array[j]);CHKERRQ(ierr);   
      ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
      ierr = MPI_Isend(array[j],ft->n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&ft2->send_reqs[j]);CHKERRQ(ierr);
      ierr = ISDestroy(&isindex);CHKERRQ(ierr);
      j++;
    }
  }
  if(ft2->n_recv) {
    for (j=0,idx=0,i=1; i<ft->n_neigh_lb; i++){
      n_cs = ft2->n_cs_comm[ft->neigh_lb[i]];
      if (n_cs>0) {
	ierr  = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
	ierr  = MPI_Irecv(&ft2->matrices[idx],n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&ft2->recv_reqs[j]);CHKERRQ(ierr);    
	idx  += n_cs*ft->n_shared_lb[i]; 
	j++;
      }	  
    }
  }
  if(ft2->n_recv) {ierr = MPI_Waitall(ft2->n_recv,ft2->recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
  if(ft2->n_send) {
    ierr = MPI_Waitall(ft2->n_send,ft2->send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    for (i=0;i<ft2->n_send;i++) {
      ierr = MatDenseRestoreArray(submat[i],&array[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&submat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(array);CHKERRQ(ierr);
    ierr = PetscFree(submat);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FETI2SetUpCoarseProblem_RBM"
/*@
   FETI2SetUpCoarseProblem_RBM - Sets up structures for assembling and computing the coarse problem.

   Input Parameter:
.  feti - the FETI context

   Level: developer

.keywords: FETI2
@*/
static PetscErrorCode FETI2SetUpCoarseProblem_RBM(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  Subdomain      sd = ft->subdomain;
  PetscMPIInt    i_mpi,sizeG,*c_displ,rankG,n_send,n_recv;
  PetscInt       k,k0,total_c_coarse,*idxm=NULL,*idxn=NULL,*idxa=NULL;
  /* nnz: array containing the number of block nonzeros in the upper triangular plus diagonal portion of each block*/
  PetscInt       i,j,idx,*nnz=NULL,size_floating,total_size_matrices=0;
  MPI_Comm       comm;
  PetscInt       *local_neighs=NULL,n_local_neighs,total_sz_fgmatrices,jdx,kdx,n_cs;
  
  PetscFunctionBegin;  
  /* in the following rankG and sizeG are related to the MPI_Comm comm */
  /* whereas rank and size are related to the MPI_Comm floatingComm */
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rankG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&sizeG);CHKERRQ(ierr);

  /* ====>>> Computing information of neighbours of neighbours */
  ierr  = PetscMalloc1(ft->n_neigh_lb-1,&ft2->n_neighs2);CHKERRQ(ierr);  /* n_neighs2[0] != ft->n_neigh_lb; not count myself */
  ierr  = PetscMalloc1(ft->n_neigh_lb-1,&ft2->send_reqs);CHKERRQ(ierr);
  ierr  = PetscMalloc1(ft->n_neigh_lb-1,&ft2->recv_reqs);CHKERRQ(ierr);
  /* n_neighs2[0] != ft->n_neigh_lb; not count myself */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Isend(&ft->n_neigh_lb,1,MPIU_INT,i_mpi,0,comm,&ft2->send_reqs[i-1]);CHKERRQ(ierr);
    ierr = MPI_Irecv(&ft2->n_neighs2[i-1],1,MPIU_INT,i_mpi,0,comm,&ft2->recv_reqs[i-1]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(ft->n_neigh_lb-1,ft2->recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(ft->n_neigh_lb-1,ft2->send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

  n_local_neighs = 0;
  for (i=1; i<ft->n_neigh_lb; i++){
    n_local_neighs  += ft2->n_neighs2[i-1];
  }
  
  ierr = PetscMalloc1(ft->n_neigh_lb-1,&ft2->neighs2);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_neighs,&ft2->neighs2[0]);CHKERRQ(ierr);
  for (i=1;i<ft->n_neigh_lb-1;i++) {
    ft2->neighs2[i] = ft2->neighs2[i-1] + ft2->n_neighs2[i-1];
  }
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Isend(ft->neigh_lb,ft->n_neigh_lb,MPIU_INT,i_mpi,0,comm,&ft2->send_reqs[i-1]);CHKERRQ(ierr);
    ierr = MPI_Irecv(ft2->neighs2[i-1],ft2->n_neighs2[i-1],MPIU_INT,i_mpi,0,comm,&ft2->recv_reqs[i-1]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(ft->n_neigh_lb-1,ft2->recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(ft->n_neigh_lb-1,ft2->send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree(ft2->send_reqs);CHKERRQ(ierr);
  ierr = PetscFree(ft2->recv_reqs);CHKERRQ(ierr);
  /* ====<<< Computing information of neighbours of neighbours */

  
  /* computing n_cs_comm that is number of rbm per subdomain and the communicator of floating structures */
  ierr = PetscMalloc1(sizeG,&ft2->n_cs_comm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&ft->n_cs,1,MPIU_INT,ft2->n_cs_comm,1,MPIU_INT,comm);CHKERRQ(ierr);
  
  /* computing displ structure for next allgatherv and total number of rbm of the system */
  ierr               = PetscMalloc1(sizeG,&ft2->displ);CHKERRQ(ierr);
  ft2->displ[0]      = 0;
  n_cs              = ft2->n_cs_comm[0];
  ft2->total_rbm     = n_cs;
  size_floating      = (n_cs>0);
  for (i=1;i<sizeG;i++){
    n_cs              = ft2->n_cs_comm[i];
    ft2->total_rbm    += n_cs;
    size_floating     += (n_cs>0);
    ft2->displ[i]      = ft2->displ[i-1] + ft2->n_cs_comm[i-1];
  }

  /* localnnz: nonzeros for my row of the coarse probem */
  if(ft->n_cs>0) {ierr = PetscMalloc1(n_local_neighs,&local_neighs);CHKERRQ(ierr);}
  n_local_neighs      = 0;
  ft2->localnnz       = 0;
  total_size_matrices = 0;
  total_sz_fgmatrices = 0;
  ft2->max_n_cs      = ft->n_cs;
  n_send              = (ft->n_neigh_lb-1)*(ft->n_cs>0);
  n_recv              = 0;
  ft2->n_send2        = 0;
  ft2->n_recv2        = 0;
  for (i=1;i<ft->n_neigh_lb;i++) {
    k0      = ft->neigh_lb[i];
    n_cs   = ft2->n_cs_comm[k0];
    /* for communicating Gs */
    n_recv              += (n_cs>0);
    total_size_matrices += n_cs*ft->n_shared_lb[i];
    /* I send */
    ft2->n_send2  += (rankG>k0)*(n_cs>0)*(ft->n_cs>0);/*send myself Fs*Gs*/
    for (j=1;j<ft->n_neigh_lb;j++) { /*send Fs*G_{my_neighs}U{k0}*/
      k      = ft->neigh_lb[j];
      n_cs  = ft2->n_cs_comm[k];
      if(k>=k0) {
	ft2->n_send2  += (n_cs>0);
      }
    }  
    /* I receive */
    for (j=0;j<ft2->n_neighs2[i-1];j++) { /*receive F_{my_neighs}*G_{my_neighs}U{neighs_neighs}*/
      k              = ft2->neighs2[i-1][j];
      n_cs          = ft2->n_cs_comm[k];
      ft2->max_n_cs = (ft2->max_n_cs > n_cs) ? ft2->max_n_cs : n_cs;
      if(rankG<=k && n_cs>0 && ft->n_cs>0) {
	ft2->n_recv2++;
	total_sz_fgmatrices += n_cs*ft->n_shared_lb[i];
	local_neighs[n_local_neighs++] = k;
      }
    }
  }
  
  ft2->n_sum_mats = 0;
  if (ft->n_cs>0) {
    ierr            = PetscSortRemoveDupsInt(&n_local_neighs,local_neighs);CHKERRQ(ierr);
    for (i=0;i<n_local_neighs;i++) {
      n_cs            = ft2->n_cs_comm[local_neighs[i]];
      ft2->localnnz   += n_cs;
      ft2->n_sum_mats ++; /* += (n_cs>0); */
    }
    ierr = PetscMalloc1(ft->n_lambda_local*ft2->localnnz,&ft2->bufferPSum);CHKERRQ(ierr);
    ierr = PetscMalloc2(ft2->n_sum_mats,&ft2->sum_mats,ft2->n_sum_mats,&ft2->i2rank);CHKERRQ(ierr);
    ierr = PetscMalloc1(ft2->localnnz,&idxn);CHKERRQ(ierr);
    ierr = PetscMalloc1(ft->n_cs,&idxm);CHKERRQ(ierr);
    /** row indices */
    jdx  = ft2->displ[rankG];
    for (i=0; i<ft->n_cs; i++) idxm[i] = i + jdx;

    for (kdx=0,k=0,i=0;i<n_local_neighs;i++) {
      k0             = local_neighs[i];
      n_cs          = ft2->n_cs_comm[k0];
      ft2->i2rank[i] = k0;
      ierr           = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,n_cs,ft2->bufferPSum+kdx,&ft2->sum_mats[i]);CHKERRQ(ierr);
      ierr           = MatSetOption(ft2->sum_mats[i],MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      kdx           += ft->n_lambda_local*n_cs;
      /** col indices */
      jdx = ft2->displ[k0];
      for (j=0;j<n_cs;j++, k++) idxn[k] = j + jdx;
    }    
    ierr = PetscFree(local_neighs);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_cs,ft2->localnnz,NULL,&ft2->local_rows_matrix);CHKERRQ(ierr);
    ierr = MatSetOption(ft2->local_rows_matrix,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  }
  
  ierr = PetscMalloc1(ft2->total_rbm,&nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(size_floating,&idxa);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_displ);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&ft2->c_count);CHKERRQ(ierr);
  ft2->c_count[0] = (ft2->n_cs_comm[0]>0);
  c_displ[0]      = 0;
  for (i=1;i<sizeG;i++) {
    c_displ[i]      = c_displ[i-1] + ft2->c_count[i-1];
    ft2->c_count[i] = (ft2->n_cs_comm[i]>0);
  }
  ierr = MPI_Allgatherv(&ft2->localnnz,(ft2->n_cs_comm[rankG]>0),MPIU_INT,idxa,ft2->c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);
  for (k0=0,k=0,i=0;i<sizeG;i++) {
    n_cs = ft2->n_cs_comm[i];
    for(j=0;j<n_cs;j++) nnz[k++] = idxa[k0];
    k0 += (n_cs>0);
  }
  ierr = PetscFree(idxa);CHKERRQ(ierr);

  /* create the "global" matrix for holding G^T*F*G */
  ierr = MatDestroy(&ft2->coarse_problem);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&ft2->coarse_problem);CHKERRQ(ierr);
  ierr = MatSetType(ft2->coarse_problem,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(ft2->coarse_problem,1);CHKERRQ(ierr);
  ierr = MatSetSizes(ft2->coarse_problem,ft2->total_rbm,ft2->total_rbm,ft2->total_rbm,ft2->total_rbm);CHKERRQ(ierr);
  ierr = MatSetOption(ft2->coarse_problem,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(ft2->coarse_problem,1,PETSC_DEFAULT,nnz);CHKERRQ(ierr);
  ierr = MatSetUp(ft2->coarse_problem);CHKERRQ(ierr);

  /* Structures for communicating matrices G with my neighbors */
  ft2->n_send = n_send;
  ft2->n_recv = n_recv;
  if(n_send) { ierr = PetscMalloc1(n_send,&ft2->send_reqs);CHKERRQ(ierr);}
  if(n_recv) {
    ierr = PetscMalloc1(total_size_matrices,&ft2->matrices);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_recv,&ft2->recv_reqs);CHKERRQ(ierr);
  }
  if(n_recv) {
    /* store received matrices in Gholder */
    ft2->n_Gholder = n_recv;
    ierr = PetscMalloc1(n_recv,&ft2->Gholder);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_recv,&ft2->neigh_holder);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*n_recv,&ft2->neigh_holder[0]);CHKERRQ(ierr);
    for (i=1;i<n_recv;i++) { 
      ft2->neigh_holder[i] = ft2->neigh_holder[i-1] + 2;
    }
    for (i=0,idx=0,k=1; k<ft->n_neigh_lb; k++){
      if (ft2->n_cs_comm[ft->neigh_lb[k]]>0) {
	ft2->neigh_holder[i][0] = ft->neigh_lb[k];
	ft2->neigh_holder[i][1] = k;
	ierr  = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_shared_lb[k],ft2->n_cs_comm[ft->neigh_lb[k]],&ft2->matrices[idx],&ft2->Gholder[i]);CHKERRQ(ierr);
	ierr  = MatSetOption(ft2->Gholder[i++],MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
	idx  += ft2->n_cs_comm[ft->neigh_lb[k]]*ft->n_shared_lb[k];
      }
    }
  }

  /* creating strucutres for computing F_local*G_neighbors */
  ierr = PetscMalloc3(sd->n*ft2->max_n_cs,&ft2->bufferRHS,sd->n*ft2->max_n_cs,&ft2->bufferX,ft->n_lambda_local*ft2->max_n_cs,&ft2->bufferG);CHKERRQ(ierr);

  ft2->n_FGholder = n_recv + (ft->n_cs>0);
  ierr            = PetscMalloc1(ft2->n_FGholder,&ft2->FGholder);CHKERRQ(ierr);
  for (i=0,k=0; k<ft->n_neigh_lb; k++) {
    n_cs = ft2->n_cs_comm[ft->neigh_lb[k]];
    if (n_cs)  {
      /* the following matrix is created using in column major order (the usual Fortran 77 manner) */
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,n_cs,NULL,&ft2->FGholder[i]);CHKERRQ(ierr);
      ierr = MatSetOption(ft2->FGholder[i++],MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  /* creating strucutres for communicating computed FG */
  if(ft2->n_send2) { ierr = PetscMalloc1(ft2->n_send2,&ft2->send2_reqs);CHKERRQ(ierr);}
  if(ft2->n_recv2) {
    ierr = PetscMalloc1(total_sz_fgmatrices,&ft2->fgmatrices);CHKERRQ(ierr);
    ierr = PetscMalloc1(ft2->n_recv2,&ft2->recv2_reqs);CHKERRQ(ierr);
  } 

  /* creating structures for assembling the matrix for the coarse problem */
  ierr            = PetscMalloc1(ft2->total_rbm,&ft2->r_coarse);CHKERRQ(ierr);
  n_cs           = ft2->n_cs_comm[0];
  total_c_coarse  = nnz[0]*(n_cs>0);
  ft2->c_count[0] = nnz[0]*(n_cs>0);
  c_displ[0]      = 0;
  idx             = n_cs;
  for (i=1;i<sizeG;i++) {
    n_cs             = ft2->n_cs_comm[i];
    total_c_coarse   += nnz[idx]*(n_cs>0);
    c_displ[i]        = c_displ[i-1] + ft2->c_count[i-1];
    ft2->c_count[i]   = nnz[idx]*(n_cs>0);
    idx              += n_cs;
  }
  ierr = PetscMalloc1(total_c_coarse,&ft2->c_coarse);CHKERRQ(ierr);
  /* gather rows and columns*/
  ierr = MPI_Allgatherv(idxm,ft->n_cs,MPIU_INT,ft2->r_coarse,ft2->n_cs_comm,ft2->displ,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(idxn,ft2->localnnz,MPIU_INT,ft2->c_coarse,ft2->c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);

  ierr = PetscFree(idxm);CHKERRQ(ierr);
  ierr = PetscFree(idxn);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscFree(c_displ);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI2_GATHER_NEIGH"
static PetscErrorCode FETIDestroy_FETI2_GATHER_NEIGH(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
   
  PetscFunctionBegin;  
  if(ft2->n_recv) { ierr = PetscFree(ft2->recv_reqs);CHKERRQ(ierr);}
  if(ft2->n_send) { ierr = PetscFree(ft2->send_reqs);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI2_RBM"
static PetscErrorCode FETIDestroy_FETI2_RBM(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  PetscInt       i;
   
  PetscFunctionBegin;  
  ierr = PetscFree3(ft2->bufferRHS,ft2->bufferX,ft2->bufferG);CHKERRQ(ierr);
  ierr = PetscFree(ft2->recv2_reqs);CHKERRQ(ierr);
  ierr = PetscFree(ft2->fgmatrices);CHKERRQ(ierr);
  ierr = PetscFree(ft2->send2_reqs);CHKERRQ(ierr);
  ierr = MatDestroy(&ft2->local_rows_matrix);CHKERRQ(ierr);  
  for (i=0;i<ft2->n_sum_mats;i++) {
    ierr = MatDestroy(&ft2->sum_mats[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(ft2->sum_mats,ft2->i2rank);CHKERRQ(ierr);
  ierr = PetscFree(ft2->bufferPSum);CHKERRQ(ierr);
  ierr = PetscFree(ft2->c_coarse);CHKERRQ(ierr);
  ierr = PetscFree(ft2->r_coarse);CHKERRQ(ierr);
  ierr = PetscFree(ft2->c_count);CHKERRQ(ierr);
  ierr = PetscFree(ft2->n_cs_comm);CHKERRQ(ierr);
  ierr = PetscFree(ft2->n_neighs2);CHKERRQ(ierr);
  ierr = PetscFree(ft2->neighs2[0]);CHKERRQ(ierr);
  ierr = PetscFree(ft2->neighs2);CHKERRQ(ierr);
  if (ft2->FGholder) {
    for (i=0;i<ft2->n_FGholder;i++) { ierr = MatDestroy(&ft2->FGholder[i]);CHKERRQ(ierr); }
    ierr = PetscFree(ft2->FGholder);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FETI2ComputeCoarseProblem_RBM"
/*@
   FETI2ComputeCoarseProblem_RBM - Once the structures needed by the
   coarse problem are built, this function computes and assembles the
   coarse problem.

   Input Parameter:
.  feti - the FETI context

   Notes: FETI2ComputeMatrixG_Private() should be called before
   calling FETI2SetUpCoarseProblem_Private().

   Level: developer

.keywords: FETI2
@*/
static PetscErrorCode FETI2ComputeCoarseProblem_RBM(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  Subdomain      sd = ft->subdomain;
  PetscMPIInt    i_mpi0,i_mpi1,sizeG,rankG;
  MPI_Comm       comm;
  IS             isindex;
  Mat            RHS,X,x,Gexpanded,*submat,aux_mat;
  PetscScalar    *pointer_vec2=NULL,*pointer_vec1=NULL,*m_pointer=NULL,*m_pointer1=NULL,**array=NULL;
  Vec            vec1,vec2;
  PetscInt       i,j,k,k0,idx,jdx,kdx,*idxm=NULL,*idxn=NULL;
  PetscInt       n_cs,delta,sz;
   
  PetscFunctionBegin;  
  /* in the following rankG and sizeG are related to the MPI_Comm comm */
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rankG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&sizeG);CHKERRQ(ierr);
  
  /* computing F_local*G_neighbors */
  ierr = PetscMalloc1(ft2->max_n_cs,&idxn);CHKERRQ(ierr);
  for (i=0;i<ft2->max_n_cs;i++) idxn[i]=i;

  for (i=0,k=0; k<ft->n_neigh_lb; k++) {
    n_cs = ft2->n_cs_comm[ft->neigh_lb[k]];
    if (n_cs>0) {
      /* the following matrix is created using in column major order (the usual Fortran 77 manner) */
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,n_cs,ft2->bufferX,&X);CHKERRQ(ierr);
      ierr = MatSetOption(X,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,n_cs,ft2->bufferRHS,&RHS);CHKERRQ(ierr);
      ierr = MatSetOption(RHS,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      if (k>0) {
	ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,n_cs,ft2->bufferG,&Gexpanded);CHKERRQ(ierr);
	ierr = MatSetOption(Gexpanded,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
	ierr = MatDenseGetArray(ft2->Gholder[i-!(ft->n_cs==0)],&m_pointer);CHKERRQ(ierr);
	ierr = MatZeroEntries(Gexpanded);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(Gexpanded,ft->n_shared_lb[k],ft->shared_lb[k],n_cs,idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(Gexpanded,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Gexpanded,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(ft2->Gholder[i-!(ft->n_cs==0)],&m_pointer);CHKERRQ(ierr);
      } else {
	Gexpanded = ft2->localG;
      }
	
      /**** RHS = B^T*Gexpanded */
      ierr = MatDenseGetArray(Gexpanded,&pointer_vec2);CHKERRQ(ierr);
      ierr = MatDenseGetArray(RHS,&pointer_vec1);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,ft->n_lambda_local,NULL,&vec2);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,sd->n,NULL,&vec1);CHKERRQ(ierr);
      for (j=0;j<n_cs;j++) {
	ierr = VecPlaceArray(vec2,(const PetscScalar*)(pointer_vec2+ft->n_lambda_local*j));CHKERRQ(ierr);
	ierr = VecPlaceArray(vec1,(const PetscScalar*)(pointer_vec1+sd->n*j));CHKERRQ(ierr);
	ierr = MatMultTranspose(ft->B_delta,vec2,sd->vec1_B);CHKERRQ(ierr);
	ierr = VecSet(vec1,0);CHKERRQ(ierr);
	ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,vec1,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
	ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,vec1,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
	ierr = VecResetArray(vec2);CHKERRQ(ierr);
	ierr = VecResetArray(vec1);CHKERRQ(ierr);
      }   
      ierr = MatDenseRestoreArray(Gexpanded,&pointer_vec2);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(RHS,&pointer_vec1);CHKERRQ(ierr);
      ierr = VecDestroy(&vec2);CHKERRQ(ierr);
      ierr = VecDestroy(&vec1);CHKERRQ(ierr);

      /**** solve system Kt*X = RHS */
      ierr = MatMatSolve(ft->F_neumann,RHS,X);CHKERRQ(ierr);

      /****  compute B*X */
      ierr = MatGetSubMatrix(X,sd->is_B_local,NULL,MAT_INITIAL_MATRIX,&x);CHKERRQ(ierr);
      ierr = MatMatMult(ft->B_delta,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&ft2->FGholder[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&x);CHKERRQ(ierr);
      ierr = MatDestroy(&RHS);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
      
      if (k>0) { ierr = MatDestroy(&Gexpanded);CHKERRQ(ierr); }
      i++;
    }
  }

  /* communicate computed FG */
  if(ft2->n_send2) { ierr = PetscMalloc2(ft2->n_send2,&submat,ft2->n_send2,&array);CHKERRQ(ierr);}
  for (delta=0,idx=0,jdx=0,i=1;i<ft->n_neigh_lb;i++) {
    kdx  = 0;
    k0   = ft->neigh_lb[i];
    ierr = PetscMPIIntCast(k0,&i_mpi0);CHKERRQ(ierr);
    /* I send */
    if (ft2->n_send2) {
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
      if (ft->n_cs) {
	if (ft2->n_cs_comm[k0]>0 && k0<rankG) {/*send myself Fs*Gs*/
	  /*>>>*/
	  ierr = MatGetSubMatrix(ft2->FGholder[kdx],isindex,NULL,MAT_INITIAL_MATRIX,&submat[idx]);CHKERRQ(ierr);
	  ierr = MatDenseGetArray(submat[idx],&array[idx]);CHKERRQ(ierr);   
	  ierr = MPI_Isend(array[idx],ft->n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi0,rankG,comm,&ft2->send2_reqs[idx]);CHKERRQ(ierr);
	  idx++;
	  /*<<<*/
	}
      	kdx++;
      }
      for (j=1;j<ft->n_neigh_lb;j++) { /*send Fs*G_{my_neighs}U{k0}*/
	k = ft->neigh_lb[j];
	if (ft2->n_cs_comm[k]>0) {
	  if(k0<=k) {
	    /*>>>*/
	    ierr = MatGetSubMatrix(ft2->FGholder[kdx],isindex,NULL,MAT_INITIAL_MATRIX,&submat[idx]);CHKERRQ(ierr);
	    ierr = MatDenseGetArray(submat[idx],&array[idx]);CHKERRQ(ierr);   
	    ierr = MPI_Isend(array[idx],ft2->n_cs_comm[k]*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi0,k,comm,&ft2->send2_reqs[idx]);CHKERRQ(ierr);
	    /*<<<*/
	    idx++;
	  }
	  kdx++;
	}
      }
      ierr = ISDestroy(&isindex);CHKERRQ(ierr);
    }
    /* I receive */   
    if(ft2->n_recv2) {
      for (j=0;j<ft2->n_neighs2[i-1];j++) { /*receive F_{my_neighs}*G_{my_neighs}U{neighs_neighs}*/
	k     = ft2->neighs2[i-1][j];
	n_cs = ft2->n_cs_comm[k];
	if(rankG<=k && n_cs) {
	  /*>>>*/
	  sz     = n_cs*ft->n_shared_lb[i];
	  ierr   = MPI_Irecv(&ft2->fgmatrices[delta],sz,MPIU_SCALAR,i_mpi0,k,comm,&ft2->recv2_reqs[jdx]);CHKERRQ(ierr);    
	  delta += sz; 
	  jdx++;
	  /*<<<*/
	}
      }
    }
  }
  if(ft2->n_recv2) {
    ierr = MPI_Waitall(ft2->n_recv2,ft2->recv2_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  }
  if(ft2->n_send2) {
    ierr = MPI_Waitall(ft2->n_send2,ft2->send2_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    for (i=0;i<ft2->n_send2;i++) {
      ierr = MatDenseRestoreArray(submat[i],&array[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&submat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(submat,array);CHKERRQ(ierr);
  }

  /** perfoming the actual multiplication G_{rankG}^T*F*G_{neigh_rankG>=rankG} */   
  if (ft->n_cs) {
    for (i=0;i<ft2->n_sum_mats;i++) { ierr = MatZeroEntries(ft2->sum_mats[i]);CHKERRQ(ierr); }
    ierr = PetscMalloc1(ft->n_lambda_local,&idxm);CHKERRQ(ierr);
    for (i=0;i<ft->n_lambda_local;i++) idxm[i] = i;
    /* sum F*G_{neigh_rankG>=rankG} */
    for (idx=0,delta=0,i=0;i<ft->n_neigh_lb;i++) {
      k0    = ft->neigh_lb[i];
      n_cs = ft2->n_cs_comm[k0];
      if (n_cs) {
	if (k0>=rankG) {	
	  ierr = PetscFindInt(k0,ft2->n_sum_mats,ft2->i2rank,&jdx);CHKERRQ(ierr);
	  ierr = MatDenseGetArray(ft2->FGholder[idx],&m_pointer);CHKERRQ(ierr);
	  ierr = MatSetValuesBlocked(ft2->sum_mats[jdx],ft->n_lambda_local,idxm,n_cs,idxn,m_pointer,ADD_VALUES);CHKERRQ(ierr);
	  ierr = MatDenseRestoreArray(ft2->FGholder[idx],&m_pointer);CHKERRQ(ierr);
	}
	idx++;
      }
      if (i) {
	for (j=0;j<ft2->n_neighs2[i-1];j++) { 
	  k     = ft2->neighs2[i-1][j];
	  n_cs = ft2->n_cs_comm[k];
	  if(rankG<=k && n_cs) {
	    ierr = PetscFindInt(k,ft2->n_sum_mats,ft2->i2rank,&jdx);CHKERRQ(ierr);
	    ierr = MatSetValuesBlocked(ft2->sum_mats[jdx],ft->n_shared_lb[i],ft->shared_lb[i],n_cs,idxn,&ft2->fgmatrices[delta],ADD_VALUES);CHKERRQ(ierr);
	    delta += n_cs*ft->n_shared_lb[i]; 
	  }
	}
      }
    }
    for (i=0;i<ft2->n_sum_mats;i++) { ierr = MatAssemblyBegin(ft2->sum_mats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); }
    for (i=0;i<ft2->n_sum_mats;i++) { ierr = MatAssemblyEnd  (ft2->sum_mats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); }

    /* multiply by G^T */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,ft2->localnnz,ft2->bufferPSum,&aux_mat);CHKERRQ(ierr);
    ierr = MatSetOption(aux_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(ft2->localG,aux_mat,MAT_REUSE_MATRIX,PETSC_DEFAULT,&ft2->local_rows_matrix);CHKERRQ(ierr);
    ierr = MatDestroy(&aux_mat);CHKERRQ(ierr);   
    ierr = PetscFree(idxm);CHKERRQ(ierr);
    ierr = PetscFree(idxn);CHKERRQ(ierr);
    
    /** local "row block" contribution to G^T*G */
    ierr = MatDenseGetArray(ft2->local_rows_matrix,&m_pointer);CHKERRQ(ierr);
  }
  
  /* assemble matrix */
  /* gather values for the coarse problem's matrix and assemble it */ 
  for (k=0,k0=0,i=0; i<sizeG; i++) {
    n_cs = ft2->n_cs_comm[i];
    if(n_cs>0){
      ierr = PetscMPIIntCast(n_cs*ft2->c_count[i],&i_mpi0);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(i,&i_mpi1);CHKERRQ(ierr);
      if(i==rankG) {
	ierr = MPI_Bcast(m_pointer,i_mpi0,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(ft2->coarse_problem,n_cs,&ft2->r_coarse[k],ft2->c_count[i],&ft2->c_coarse[k0],m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(ft2->local_rows_matrix,&m_pointer);CHKERRQ(ierr);
      } else {
	ierr = PetscMalloc1(i_mpi0,&m_pointer1);CHKERRQ(ierr);	  
	ierr = MPI_Bcast(m_pointer1,i_mpi0,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(ft2->coarse_problem,n_cs,&ft2->r_coarse[k],ft2->c_count[i],&ft2->c_coarse[k0],m_pointer1,INSERT_VALUES);CHKERRQ(ierr);
	ierr = PetscFree(m_pointer1);CHKERRQ(ierr);
      }
      k0 += ft2->c_count[i];
      k  += n_cs;
    }
  }
  ierr = MatAssemblyBegin(ft2->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ft2->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2FactorizeCoarseProblem_Private"
/*@
  FETI2FactorizeCoarseProblem_Private - Factorizes the coarse problem. 

  Input Parameter:
  .  ft - the FETI context

   Notes: 
   FETI2SetUpCoarseProblem_Private() must be called before calling FETI2FactorizeCoarseProblem_Private().

   Level: developer

.keywords: FETI2

.seealso: FETI2SetUpCoarseProblem_Private()
@*/
static PetscErrorCode FETI2FactorizeCoarseProblem_Private(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  PC             pc;
  
  PetscFunctionBegin;
  if(!ft2->coarse_problem) SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: FETI2SetUpCoarseProblem_Private() must be first called");

  /* factorize the coarse problem */
  if(!ft2->ksp_coarse) {
    ierr = KSPCreate(PETSC_COMM_SELF,&ft2->ksp_coarse);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ft2->ksp_coarse,(PetscObject)ft2,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ft2,(PetscObject)ft2->ksp_coarse);CHKERRQ(ierr);
    ierr = KSPSetType(ft2->ksp_coarse,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(ft2->ksp_coarse,"feti2_pc_coarse_");CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(ft2->coarse_problem,"feti2_pc_coarse_");CHKERRQ(ierr);
    ierr = KSPGetPC(ft2->ksp_coarse,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ft2->ksp_coarse);CHKERRQ(ierr);
    ierr = KSPSetOperators(ft2->ksp_coarse,ft2->coarse_problem,ft2->coarse_problem);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
  } else {
    ierr = KSPGetPC(ft2->ksp_coarse,&pc);CHKERRQ(ierr);
    ierr = KSPSetOperators(ft2->ksp_coarse,ft2->coarse_problem,ft2->coarse_problem);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);

  }
  ierr = KSPSetUp(ft2->ksp_coarse);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&ft2->F_coarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2ApplyCoarseProblem_Private"
/*@
   FETI2ApplyCoarseProblem_Private - Applies the operation G*(G^T*F*G)^{-1} to a vector  

   Input Parameter:
.  ft       - the FETI context
.  v  - the input vector (in this case v is a local copy of the global assembled vector)

   Output Parameter:
.  r  - the output vector. 

   Level: developer

.keywords: FETI2

.seealso: FETI2ApplyCoarseProblem_Private()
@*/
static PetscErrorCode FETI2ApplyCoarseProblem_Private(FETI ft,Vec v,Vec r)
{
  PetscErrorCode     ierr;
  FETI_2             *ft2 = (FETI_2*)ft->data;
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
  if(!ft2->F_coarse) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: FETI2FactorizeCoarseProblem_Private() must be first called");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* apply (G^T*G)^{-1}: compute v_rbm = (G^T*G)^{-1}*v */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft2->total_rbm,&v_rbm);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(ft2->F_coarse,25,0);CHKERRQ(ierr);
#endif
  ierr = MatSolve(ft2->F_coarse,v,v_rbm);CHKERRQ(ierr);

  /* apply G: compute r = G*v_rbm */
  ierr = VecUnAsmGetLocalVector(r,&r_local);CHKERRQ(ierr);
  ierr = PetscMalloc1(ft2->max_n_cs,&indices);CHKERRQ(ierr);
  /** mulplying by localG for the current processor */
  if(ft->n_cs) {
    for (i=0;i<ft->n_cs;i++) indices[i] = ft2->displ[rank] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_cs,indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
    ierr = VecGetSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = MatMult(ft2->localG,v0,r_local);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = ISDestroy(&subset);CHKERRQ(ierr);
  } else {
    ierr = VecSet(r_local,0.0);CHKERRQ(ierr);
  }
  /** multiplying by localG for the other processors */
  for (j=0;j<ft2->n_Gholder;j++) {
    idx0 = ft2->neigh_holder[j][0];
    idx1 = ft2->neigh_holder[j][1];   
    for (i=0;i<ft2->n_cs_comm[idx0];i++) indices[i] = ft2->displ[idx0] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft2->n_cs_comm[idx0],indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
    ierr = VecGetSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_shared_lb[idx1],&vec_holder);CHKERRQ(ierr);
    ierr = MatMult(ft2->Gholder[j],v0,vec_holder);CHKERRQ(ierr);
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
#define __FUNCT__ "FETI2ComputeInitialCondition_RBM"
/*@
   FETI2ComputeInitialCondition_RBM - Computes initial condition
   for the interface problem. 

   Input Parameter:
.  ft - the FETI context

   Level: developer

.keywords: FETI2

@*/
static PetscErrorCode FETI2ComputeInitialCondition_RBM(FETI ft)
{
  PetscErrorCode    ierr;
  FETI_2            *ft2 = (FETI_2*)ft->data;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  Vec               lambda_local,localv;
  
  PetscFunctionBegin;
  ft2  = (FETI_2*)ft->data;
  comm = PetscObjectComm((PetscObject)ft);
  ierr = VecUnAsmGetLocalVectorRead(ft->d,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft2->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft2->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecUnAsmRestoreLocalVectorRead(ft->d,lambda_local);CHKERRQ(ierr);
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,ft2->n_cs_comm,ft2->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETI2ApplyCoarseProblem_Private(ft,asm_e,ft->lambda_global);CHKERRQ(ierr);
  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2ComputeInitialCondition_NOCOARSE"
/*@
   FETI2ComputeInitialCondition_NOCOARSE - Computes initial condition
   for the interface problem. 

   Input Parameter:
.  ft - the FETI context

   Level: developer

.keywords: FETI2

@*/
static PetscErrorCode FETI2ComputeInitialCondition_NOCOARSE(FETI ft)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = VecSet(ft->lambda_global, 0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2Project_RBM"
/*@
   FETI2Project_RBM - Performs the projection step of FETI2.

   Input Parameter:
.  ft        - the FETI context
.  g_global  - the vector to project
.  y         - the projected vector

   Level: developer

.keywords: FETI2

@*/
PetscErrorCode FETI2Project_RBM(void* ft_ctx, Vec g_global, Vec y)
{
  PetscErrorCode    ierr;
  FETI              ft; 
  FETI_2            *ft2;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  Vec               lambda_local,localv,y_aux;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft_ctx,FETI_CLASSID,1);
  PetscValidHeaderSpecific(g_global,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ft   = (FETI)ft_ctx;
  ft2  = (FETI_2*)ft->data;
  comm = PetscObjectComm((PetscObject)ft);
  if (y == g_global) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use g_global == y");
  ierr = VecUnAsmGetLocalVectorRead(g_global,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft2->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft2->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,ft2->n_cs_comm,ft2->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(g_global,lambda_local);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETI2ApplyCoarseProblem_Private(ft,asm_e,y);CHKERRQ(ierr);

  ierr = VecDuplicate(y,&y_aux);CHKERRQ(ierr);  
  ierr = MultFv_Private(ft,y,y_aux);CHKERRQ(ierr);

  ierr = VecWAXPY(y,-1,y_aux,g_global);CHKERRQ(ierr);

  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  ierr = VecDestroy(&y_aux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI2ReProject_RBM"
/*@
   FETI2ReProject_RBM - Performs the re-projection step of FETI2.

   Input Parameter:
.  ft        - the FETI context
.  g_global  - the vector to project
.  y         - the projected vector

   Level: developer

.keywords: FETI2

@*/
PetscErrorCode FETI2ReProject_RBM(void* ft_ctx, Vec g_global, Vec y)
{
  PetscErrorCode    ierr;
  FETI              ft; 
  FETI_2            *ft2;
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
  ft2  = (FETI_2*)ft->data;
  comm = PetscObjectComm((PetscObject)ft);
  if (y == g_global) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use g_global == y");
  
  ierr = MultFv_Private(ft,g_global,y);CHKERRQ(ierr);
  
  ierr = VecUnAsmGetLocalVectorRead(y,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft2->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft2->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,ft2->n_cs_comm,ft2->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(y,lambda_local);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETI2ApplyCoarseProblem_Private(ft,asm_e,y);CHKERRQ(ierr);
  ierr = VecAYPX(y,-1,g_global);CHKERRQ(ierr);

  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
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
                                 -feti2_pc_coarse_pc_factor_mat_solver_package mumps   \
                                 -feti2_pc_coarse_mat_mumps_icntl_7 2";
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

