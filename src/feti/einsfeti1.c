#include <../src/feti/einsfeti1.h>
#include <einsksp.h>
#include <einssys.h>


/* private functions*/
static PetscErrorCode FETI1BuildLambdaAndB_Private(FETI);
static PetscErrorCode FETI1SetUpNeumannSolver_Private(FETI);
static PetscErrorCode FETI1ComputeMatrixGandRhsE_Private(FETI);
static PetscErrorCode FETI1BuildInterfaceProblem_Private(FETI);
static PetscErrorCode FETIDestroy_FETI1(FETI);
static PetscErrorCode FETISetUp_FETI1(FETI);
static PetscErrorCode FETI1DestroyMatF_Private(Mat);
static PetscErrorCode FETI1MatMult_Private(Mat,Vec,Vec);
static PetscErrorCode FETISetFromOptions_FETI1(PetscOptions*,FETI);
static PetscErrorCode FETI1SetUpCoarseProblem_Private(FETI);
static PetscErrorCode FETI1FactorizeCoarseProblem_Private(FETI);
static PetscErrorCode FETI1ApplyCoarseProblem_Private(FETI,Vec,Vec);
static PetscErrorCode FETI1ComputeInitialCondition_Private(FETI);
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
  ierr = MatDestroy(&ft1->rbm);CHKERRQ(ierr);
  ierr = MatDestroy(&ft1->localG);CHKERRQ(ierr);
  ierr = VecDestroy(&ft1->local_e);CHKERRQ(ierr);
  if(ft1->neigh_holder) {
    ierr = PetscFree(ft1->neigh_holder[0]);CHKERRQ(ierr);
    ierr = PetscFree(ft1->neigh_holder);CHKERRQ(ierr);
  }
  if(ft1->displ) { ierr = PetscFree(ft1->displ);CHKERRQ(ierr);}
  if(ft1->count_rbm) { ierr = PetscFree(ft1->count_rbm);CHKERRQ(ierr);}
  for (i=0;i<ft1->n_Gholder;i++) {
    ierr = MatDestroy(&ft1->Gholder[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(ft1->Gholder);CHKERRQ(ierr);
  if(ft1->matrices) { ierr = PetscFree(ft1->matrices);CHKERRQ(ierr);}
  ierr = PetscFree(ft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* #undef __FUNCT__ */
/* #define __FUNCT__ "FETI1ConvergedProjectedResidual" */
/* /\*@ */
/*    FETI1ConvergedProjectedResidual - Checks convergence of the */
/*    iterface problem by comparing the L2 norm of the projected */
/*    residual. */

/*    Input Parameter: */
/* .  ksp    - the ksp context */
/* .  n      - the iteration number */
/* .  rnorm  - the L2 norm of the projected residual */
/* .  reason - reason a Krylov method was said to have converged or diverged */
/* .  ctx    - optional convergence context */
 
/*    Level: intermediate */

/* .keywords: FETI */

/* @*\/ */
/* PetscErrorCode  FETI1ConvergedProjectedResidual(KSP ksp,PetscInt n,PetscReal rnorm,KSPConvergedReason *reason,void *ctx) */
/* { */
/*   PetscErrorCode         ierr; */
  
/*   PetscFunctionBegin; */

  
  
/*   PetscFunctionReturn(0); */
/* } */


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETI1"
/*@
   FETISetUp_FETI1 - Prepares the structures needed by the FETI-1 solver.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETISetUp_FETI1(FETI ft)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = FETIScalingSetUp(ft);CHKERRQ(ierr);
  ierr = FETI1BuildLambdaAndB_Private(ft);CHKERRQ(ierr);
  ierr = FETI1SetUpNeumannSolver_Private(ft);CHKERRQ(ierr);
  ierr = FETI1ComputeMatrixGandRhsE_Private(ft);CHKERRQ(ierr);
  ierr = FETI1BuildInterfaceProblem_Private(ft);CHKERRQ(ierr);
  ierr = FETIBuildInterfaceKSP(ft);CHKERRQ(ierr);
  /* set projection in ksp */
  ierr = KSPSetProjection(ft->ksp_interface,FETI1Project_RBM,(void*)ft);CHKERRQ(ierr);
  ierr = KSPSetReProjection(ft->ksp_interface,FETI1Project_RBM,(void*)ft);CHKERRQ(ierr);
  ierr = FETI1SetUpCoarseProblem_Private(ft);CHKERRQ(ierr);
  ierr = FETI1FactorizeCoarseProblem_Private(ft);CHKERRQ(ierr);
  ierr = FETI1ComputeInitialCondition_Private(ft);CHKERRQ(ierr);

#if defined(FETI_DEBUG)
  {
    FETI_1            *ft1 = (FETI_1*)ft->data;
    Vec               g_global,y_g,y_g2,col,asm_e,localv;
    PetscMPIInt       rank;
    PetscScalar       *rbuff;
    const PetscScalar *sbuff;
    MPI_Comm          comm;
    ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

    /* TEST A */
    PetscPrintf(PETSC_COMM_WORLD,"\n==================================================\n");
    PetscPrintf(PETSC_COMM_WORLD,"\n                    TEST A \n");
    PetscPrintf(PETSC_COMM_WORLD,"==================================================\n");

    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_lambda_local,&col);CHKERRQ(ierr);
    ierr = VecDuplicate(ft->lambda_global,&g_global);CHKERRQ(ierr);
    ierr = VecDuplicate(ft->lambda_global,&y_g);CHKERRQ(ierr);
    ierr = VecDuplicate(ft->lambda_global,&y_g2);CHKERRQ(ierr);
    ierr = VecSet(g_global,0.0);CHKERRQ(ierr);
    ierr = VecSet(col,0.0);CHKERRQ(ierr);
    if(rank==1){
      ierr = MatGetColumnVector(ft1->localG,col,0);CHKERRQ(ierr);
      MatView(ft1->localG,PETSC_VIEWER_STDOUT_SELF);
    }
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);   
    ierr = VecScatterBegin(ft->l2g_lambda,col,g_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ft->l2g_lambda,col,g_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->total_rbm,&asm_e);CHKERRQ(ierr);
    ierr = VecScatterBegin(ft->l2g_lambda,g_global,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(ft->l2g_lambda,g_global,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    if (ft1->n_rbm) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->n_rbm,&localv);CHKERRQ(ierr);
      ierr = MatMultTranspose(ft1->localG,ft->lambda_local,localv);CHKERRQ(ierr);   
      ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
    }
    ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
    ierr = MPI_Allgatherv(sbuff,ft1->n_rbm,MPIU_SCALAR,rbuff,ft1->count_rbm,ft1->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
    ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
    if (ft1->n_rbm) {
      ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
      ierr = VecDestroy(&localv);CHKERRQ(ierr);
    }

    ierr = FETI1ApplyCoarseProblem_Private(ft,asm_e,y_g2);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"\n-------------------->>>>>       GLOBAL_VECTOR \n");
    VecView(g_global,PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,"\n-------------------->>>>>       Result of G*(G^T*G)^-1*G^T*(GLOBAL_VECTOR) \n");
    VecView(y_g2,PETSC_VIEWER_STDOUT_WORLD);
    
    /* TEST B */
    PetscPrintf(PETSC_COMM_WORLD,"\n=================================================================================\n");
    PetscPrintf(PETSC_COMM_WORLD,"\n                    TEST B: Result of ( I - G*(G^T*G)^-1*G^T ) * GLOBAL_VECTOR \n");
    PetscPrintf(PETSC_COMM_WORLD,"===================================================================================\n");
    ierr = FETI1Project_RBM(ft,g_global,y_g);CHKERRQ(ierr);
    VecView(y_g,PETSC_VIEWER_STDOUT_WORLD);
    ierr = VecDestroy(&g_global);CHKERRQ(ierr);
    ierr = VecDestroy(&y_g);CHKERRQ(ierr);
    ierr = VecDestroy(&y_g2);CHKERRQ(ierr);
    ierr = VecDestroy(&col);CHKERRQ(ierr);
  }
#endif
  
  ierr = KSPSetTolerances(ft->ksp_interface,1e-8,0,PETSC_DEFAULT,1000);CHKERRQ(ierr);
  ierr = KSPSolve(ft->ksp_interface,ft->d,ft->lambda_global);CHKERRQ(ierr);
  
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
static PetscErrorCode FETISetFromOptions_FETI1(PetscOptions *PetscOptionsObject,FETI ft)
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
  feti1->n_rbm                 = 0;
  feti1->total_rbm             = 0;
  feti1->max_n_rbm             = 0;
  feti1->displ                 = 0;
  feti1->count_rbm             = 0;
  
  /* function pointers */
  ft->ops->setup               = FETISetUp_FETI1;
  ft->ops->destroy             = FETIDestroy_FETI1;
  ft->ops->setfromoptions      = FETISetFromOptions_FETI1;
  ft->ops->view                = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "FETI1BuildLambdaAndB_Private"
/*@
   FETI1BuildLambdaAndB_Private - Computes the B operator and the vector lambda of 
   the interface problem.

   Input Parameters:
.  ft - the FETI context

   Notes: 
   In a future this rutine could be moved to the FETI class.

   Level: developer
   
@*/
static PetscErrorCode FETI1BuildLambdaAndB_Private(FETI ft)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  IS                IS_l2g_lambda;
  IS                subset,subset_mult,subset_n;
  PetscBool         fully_redundant;
  PetscInt          i,j,s,n_boundary_dofs,n_global_lambda,partial_sum,up;
  PetscInt          cum,n_lambda_local,n_lambda_for_dof,dual_size,n_neg_values,n_pos_values;
  PetscMPIInt       rank;
  PetscInt          *dual_dofs_boundary_indices,*aux_local_numbering_1;
  const PetscInt    *aux_global_numbering,*indices;
  PetscInt          *aux_sums,*cols_B_delta,*l2g_indices;
  PetscScalar       *array,*vals_B_delta,*vals_B_Ddelta;
  PetscInt          *aux_local_numbering_2;
  PetscScalar       scalar_value;
  Subdomain         sd = ft->subdomain;
  const PetscScalar *Warray;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  if(!ft->Wscaling) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: FETIScalingSetUp must be first called");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Default type of lagrange multipliers is non-redundant */
  fully_redundant = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-feti_fullyredundant",&fully_redundant,NULL);CHKERRQ(ierr);

  /* Evaluate local and global number of lagrange multipliers */
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  n_lambda_local = 0;
  partial_sum = 0;
  n_boundary_dofs = 0;
  dual_size = sd->n_B;
  ierr = PetscMalloc1(dual_size,&dual_dofs_boundary_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_2);CHKERRQ(ierr);
  
  ierr = ISGetIndices(sd->is_B_local,&indices);CHKERRQ(ierr); 
  ierr = VecGetArray(sd->vec1_B,&array);CHKERRQ(ierr);
  for (i=0;i<dual_size;i++){
    j = sd->count[i]; /* RECALL: sd->count[i] does not count myself */
    n_boundary_dofs++;
    if (fully_redundant) {
      /* fully redundant set of lagrange multipliers */
      n_lambda_for_dof = (j*(j+1))/2;
    } else {
      n_lambda_for_dof = j;
    }
    n_lambda_local += j;
    /* needed to evaluate global number of lagrange multipliers */
    array[i]=(1.0*n_lambda_for_dof)/(j+1.0); /* already scaled for the next global sum */
    /* store some data needed */
    dual_dofs_boundary_indices[partial_sum] = n_boundary_dofs-1;
    aux_local_numbering_1[partial_sum] = indices[i];
    aux_local_numbering_2[partial_sum] = n_lambda_for_dof;
    partial_sum++;
  }
  ierr = VecRestoreArray(sd->vec1_B,&array);CHKERRQ(ierr);
  ierr = ISRestoreIndices(sd->is_B_local,&indices);CHKERRQ(ierr);

  /* compute ft->n_lambda */
  ierr = VecSet(sd->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->global_to_B,sd->vec1_B,sd->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->global_to_B,sd->vec1_B,sd->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSum(sd->vec1_global,&scalar_value);CHKERRQ(ierr);
  ft->n_lambda = (PetscInt)PetscRealPart(scalar_value);
  
  /* compute global ordering of lagrange multipliers and associate l2g map */
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_1,PETSC_COPY_VALUES,&subset_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(sd->mapping,subset_n,&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = PetscFree(aux_local_numbering_1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_2,PETSC_OWN_POINTER,&subset_mult);CHKERRQ(ierr);
  ierr = ISSubsetNumbering(subset,subset_mult,&i,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);
  if (i != ft->n_lambda) {
    SETERRQ3(comm,PETSC_ERR_PLIB,"Error in %s: global number of multipliers mismatch! (%d!=%d)\n",__FUNCT__,ft->n_lambda,i);
  }
  ft->n_lambda_local = n_lambda_local;
  /* Compute B_delta (local actions) */
  ierr = PetscMalloc1(sd->n_neigh,&aux_sums);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&l2g_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&vals_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&vals_B_Ddelta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&cols_B_delta);CHKERRQ(ierr);
  ierr = ISGetIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ft->Wscaling,&Warray);CHKERRQ(ierr);
  n_global_lambda=0;
  partial_sum=0;
  cum = 0;
  for (i=0;i<dual_size;i++) {
    n_global_lambda = aux_global_numbering[cum];
    j = sd->count[i]; /* "sd->count[aux_local_numbering_1[i]]": aux_local_numbering_1[i] primal dof number of the boundary */
    aux_sums[0]=0;
    for (s=1;s<j;s++) {
      aux_sums[s]=aux_sums[s-1]+j-s+1;
    }
    n_neg_values = 0;
    
    while(n_neg_values < j && sd->neighbours_set[i][n_neg_values] < rank){
      n_neg_values++;
    }
    
    n_pos_values = j - n_neg_values;
    if (fully_redundant) {
      for (s=0;s<n_neg_values;s++) {
	up = dual_dofs_boundary_indices[i];
        l2g_indices    [partial_sum+s]=aux_sums[s]+n_neg_values-s-1+n_global_lambda;
        cols_B_delta   [partial_sum+s]=up;
        vals_B_delta   [partial_sum+s]=-1.0;
	vals_B_Ddelta  [partial_sum+s]=-Warray[up];
      }
      for (s=0;s<n_pos_values;s++) {
	up = dual_dofs_boundary_indices[i];
        l2g_indices    [partial_sum+s+n_neg_values]=aux_sums[n_neg_values]+s+n_global_lambda;
        cols_B_delta   [partial_sum+s+n_neg_values]=up;
        vals_B_delta   [partial_sum+s+n_neg_values]=1.0;
	vals_B_Ddelta  [partial_sum+s+n_neg_values]=Warray[up];	
      }
      partial_sum += j;
    } else {
      /* l2g_indices and default cols and vals of B_delta */
      up = dual_dofs_boundary_indices[i];
      for (s=0;s<j;s++) {
        l2g_indices    [partial_sum+s]=n_global_lambda+s;
        cols_B_delta   [partial_sum+s]=up;
        vals_B_delta   [partial_sum+s]=0.0;
	vals_B_Ddelta  [partial_sum+s]=0.0;	
      }
      /* B_delta */
      if ( n_neg_values > 0 ) { /* there's a rank next to me to the left */
        vals_B_delta   [partial_sum+n_neg_values-1]=-1.0;
	vals_B_Ddelta  [partial_sum+n_neg_values-1]=-Warray[up];	
      }
      if ( n_neg_values < j ) { /* there's a rank next to me to the right */
        vals_B_delta   [partial_sum+n_neg_values]=1.0;
	vals_B_Ddelta  [partial_sum+n_neg_values]=Warray[up];
      }
      partial_sum += j;
    }
    cum += aux_local_numbering_2[i];
  }
  ierr = VecRestoreArrayRead(ft->Wscaling,&Warray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_mult);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = PetscFree(aux_sums);CHKERRQ(ierr);
  ierr = PetscFree(dual_dofs_boundary_indices);CHKERRQ(ierr);

  /* Local to global mapping for lagrange multipliers */
  ierr = VecCreate(PETSC_COMM_SELF,&ft->lambda_local);CHKERRQ(ierr);
  ierr = VecSetSizes(ft->lambda_local,n_lambda_local,n_lambda_local);CHKERRQ(ierr);
  ierr = VecSetType(ft->lambda_local,VECSEQ);CHKERRQ(ierr);
  ierr = VecCreate(comm,&ft->lambda_global);CHKERRQ(ierr);
  ierr = VecSetSizes(ft->lambda_global,PETSC_DECIDE,ft->n_lambda);CHKERRQ(ierr);
  ierr = VecSetType(ft->lambda_global,VECMPI);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_lambda_local,l2g_indices,PETSC_OWN_POINTER,&IS_l2g_lambda);CHKERRQ(ierr);
  ierr = VecScatterCreate(ft->lambda_local,(IS)0,ft->lambda_global,IS_l2g_lambda,&ft->l2g_lambda);CHKERRQ(ierr);
  /* create local to global mapping and neighboring information for lambda */
  ierr = ISLocalToGlobalMappingCreate(comm,1,n_lambda_local,l2g_indices,PETSC_COPY_VALUES,&ft->mapping_lambda);
  ierr = ISLocalToGlobalMappingGetInfo(ft->mapping_lambda,&(ft->n_neigh_lb),&(ft->neigh_lb),&(ft->n_shared_lb),&(ft->shared_lb));CHKERRQ(ierr);
  ierr = ISDestroy(&IS_l2g_lambda);CHKERRQ(ierr);
  /* Create local part of B_delta */
  ierr = MatCreate(PETSC_COMM_SELF,&ft->B_delta);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&ft->B_Ddelta);CHKERRQ(ierr);
  ierr = MatSetSizes(ft->B_delta,n_lambda_local,sd->n_B,n_lambda_local,sd->n_B);CHKERRQ(ierr);
  ierr = MatSetSizes(ft->B_Ddelta,n_lambda_local,sd->n_B,n_lambda_local,sd->n_B);CHKERRQ(ierr);
  ierr = MatSetType(ft->B_delta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetType(ft->B_Ddelta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(ft->B_delta,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(ft->B_Ddelta,1,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(ft->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(ft->B_Ddelta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<n_lambda_local;i++) {
    ierr = MatSetValue(ft->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(ft->B_Ddelta,i,cols_B_delta[i],vals_B_Ddelta[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(ft->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (ft->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ft->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (ft->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(vals_B_delta);CHKERRQ(ierr);
  ierr = PetscFree(vals_B_Ddelta);CHKERRQ(ierr);
  ierr = PetscFree(cols_B_delta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


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
  ierr = MatShellGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = FETIDestroy(&mat_ctx->ft);CHKERRQ(ierr);
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
  FETI_1       *ft1;
  Subdomain    sd;
  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(F,(void**)&mat_ctx);CHKERRQ(ierr);
  ft   = mat_ctx->ft;
  ft1  = (FETI_1*)ft->data;
  sd   = ft->subdomain;
  /* Application of B_delta^T */
  ierr = VecScatterBegin(ft->l2g_lambda,lambda_global,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,lambda_global,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(ft->B_delta,ft->lambda_local,sd->vec1_B);CHKERRQ(ierr);
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* Application of the already factorized pseudo-inverse */
  ierr = MatMumpsSetIcntl(ft1->F_neumann,25,0);CHKERRQ(ierr);
  ierr = MatSolve(ft1->F_neumann,sd->vec1_N,sd->vec2_N);CHKERRQ(ierr);
  /* Application of B_delta */
  ierr = VecScatterBegin(sd->N_to_B,sd->vec2_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec2_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(ft->B_delta,sd->vec1_B,ft->lambda_local);CHKERRQ(ierr);
  /** Communication with other processes is performed for the following operation */
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ft->l2g_lambda,ft->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,ft->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  Subdomain      sd = ft->subdomain;
  PetscInt       rank;
  MPI_Comm       comm;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  /* Create the MatShell for F */
  ierr = FETICreateFMat(ft,(void (*)(void))FETI1MatMult_Private,(void (*)(void))FETI1DestroyMatF_Private);CHKERRQ(ierr);
  /* Creating vector d for the interface problem */
  ierr = MatCreateVecs(ft->F,NULL,&ft->d);CHKERRQ(ierr);
  /** Application of the already factorized pseudo-inverse */
  ierr = MatMumpsSetIcntl(ft1->F_neumann,25,0);CHKERRQ(ierr);
  ierr = MatSolve(ft1->F_neumann,sd->localRHS,sd->vec1_N);CHKERRQ(ierr);
  /** Application of B_delta */
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(ft->B_delta,sd->vec1_B,ft->lambda_local);CHKERRQ(ierr);
  /*** Communication with other processes is performed for the following operation */
  ierr = VecSet(ft->d,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ft->l2g_lambda,ft->lambda_local,ft->d,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,ft->lambda_local,ft->d,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1ComputeMatrixGandRhsE_Private"
/*@
   FETI1ComputeMatrixGandRhsE_Private - Computes the local matrix
   G=B*R, where R are the Rigid Body Modes, and the rhs term e=R^T*f from
   the interface problem.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI1ComputeMatrixGandRhsE_Private(FETI ft)
{
  PetscErrorCode ierr;
  Subdomain      sd = ft->subdomain;
  PetscInt       rank;
  MPI_Comm       comm;
  Mat            x; 
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  ierr   = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr   = MatDestroy(&ft1->localG);CHKERRQ(ierr);
  ierr   = VecDestroy(&ft1->local_e);CHKERRQ(ierr);
  /* get number of rigid body modes */
  ierr   = MatMumpsGetInfog(ft1->F_neumann,28,&ft1->n_rbm);CHKERRQ(ierr);
  if(ft1->n_rbm){
    /* Compute rigid body modes */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,ft1->n_rbm,NULL,&ft1->rbm);CHKERRQ(ierr);
    ierr = MatDuplicate(ft1->rbm,MAT_DO_NOT_COPY_VALUES,&x);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(ft1->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(ft1->rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMumpsSetIcntl(ft1->F_neumann,25,-1);CHKERRQ(ierr);
    ierr = MatMatSolve(ft1->F_neumann,x,ft1->rbm);CHKERRQ(ierr);
    ierr = MatDestroy(&x);CHKERRQ(ierr);

    /* compute matrix localG */
    ierr = MatGetSubMatrix(ft1->rbm,sd->is_B_local,NULL,MAT_INITIAL_MATRIX,&x);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,ft1->n_rbm,NULL,&ft1->localG);CHKERRQ(ierr);
    ierr = MatMatMult(ft->B_Ddelta,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&ft1->localG);CHKERRQ(ierr);    

    /* compute matrix local_e */
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->n_rbm,&ft1->local_e);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft1->rbm,sd->localRHS,ft1->local_e);CHKERRQ(ierr);
    ierr = MatDestroy(&x);CHKERRQ(ierr);
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
  FETI_1         *ft1 = (FETI_1*)ft->data;
  Subdomain      sd = ft->subdomain;
  
  PetscFunctionBegin;
#if !defined(PETSC_HAVE_MUMPS)
    SETERRQ(PetscObjectComm((PetscObject)ft),1,"EINS only supports MUMPS for the solution of the Neumann problem");
#endif
  if (!ft->ksp_neumann) {
    ierr = KSPCreate(PETSC_COMM_SELF,&ft->ksp_neumann);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ft->ksp_neumann,(PetscObject)ft,1);CHKERRQ(ierr);
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
    ierr = PCFactorGetMatrix(pc,&ft1->F_neumann);CHKERRQ(ierr);
    /* sequential ordering */
    ierr = MatMumpsSetIcntl(ft1->F_neumann,7,2);CHKERRQ(ierr);
    /* Null row pivot detection */
    ierr = MatMumpsSetIcntl(ft1->F_neumann,24,1);CHKERRQ(ierr);
    /* threshhold for row pivot detection */
    ierr = MatMumpsSetCntl(ft1->F_neumann,3,1.e-6);CHKERRQ(ierr);

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
   FETI1SetUpCoarseProblem_Private - It mainly configures the coarse problem and factorizes it.

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
  PetscMPIInt    i_mpi,i_mpi1,sizeG,*c_displ,*c_count,n_recv,n_send,rankG;
  PetscInt       k,k0,*c_coarse,*r_coarse,total_c_coarse,*idxm=NULL,*idxn=NULL;
  /* nnz: array containing the number of block nonzeros in the upper triangular plus diagonal portion of each block*/
  PetscInt       i,j,idx,*n_rbm_comm,*nnz,size_floating,total_size_matrices=0,localnnz=0;
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
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rankG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&sizeG);CHKERRQ(ierr);
  
  /* computing n_rbm_comm that is number of rbm per subdomain */
  ierr = PetscMalloc1(sizeG,&n_rbm_comm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&ft1->n_rbm,1,MPIU_INT,n_rbm_comm,1,MPIU_INT,comm);CHKERRQ(ierr);

  /* computing displ structure for next allgatherv and total number of rbm of the system */
  ierr               = PetscMalloc1(sizeG,&ft1->displ);CHKERRQ(ierr);
  ierr               = PetscMalloc1(sizeG,&ft1->count_rbm);CHKERRQ(ierr);
  ft1->displ[0]      = 0;
  ft1->count_rbm[0]  = n_rbm_comm[0];
  ft1->total_rbm     = n_rbm_comm[0];
  size_floating      = (n_rbm_comm[0]>0);
  for (i=1;i<sizeG;i++){
    ft1->total_rbm    += n_rbm_comm[i];
    size_floating     += (n_rbm_comm[i]>0);
    ft1->count_rbm[i]  = n_rbm_comm[i];
    ft1->displ[i]      = ft1->displ[i-1] + ft1->count_rbm[i-1];
  }

  /* localnnz: nonzeros for my row of the coarse probem */
  localnnz            = ft1->n_rbm;
  total_size_matrices = 0;
  ft1->max_n_rbm      = ft1->n_rbm;
  n_send              = (ft->n_neigh_lb-1)*(ft1->n_rbm>0);
  n_recv              = 0;
  if(ft1->n_rbm) {
    for (i=1;i<ft->n_neigh_lb;i++){
      i_mpi                = n_rbm_comm[ft->neigh_lb[i]];
      n_recv              += (i_mpi>0);
      ft1->max_n_rbm       = (ft1->max_n_rbm > i_mpi) ? ft1->max_n_rbm : i_mpi;
      total_size_matrices += i_mpi*ft->n_shared_lb[i];
      localnnz            += i_mpi*(ft->neigh_lb[i]>rankG);
    }
  } else {
    for (i=1;i<ft->n_neigh_lb;i++){
      i_mpi                = n_rbm_comm[ft->neigh_lb[i]];
      n_recv              += (i_mpi>0);
      ft1->max_n_rbm       = (ft1->max_n_rbm > i_mpi) ? ft1->max_n_rbm : i_mpi;
      total_size_matrices += i_mpi*ft->n_shared_lb[i];
    }
  }

  ierr = PetscMalloc1(ft1->total_rbm,&nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(size_floating,&idxm);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_displ);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_count);CHKERRQ(ierr);
  c_count[0]     = (n_rbm_comm[0]>0);
  c_displ[0]     = 0;
  for (i=1;i<sizeG;i++) {
    c_displ[i] = c_displ[i-1] + c_count[i-1];
    c_count[i] = (n_rbm_comm[i]>0);
  }
  ierr = MPI_Allgatherv(&localnnz,(n_rbm_comm[rankG]>0),MPIU_INT,idxm,c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);
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
  ierr = PetscMalloc1(total_size_matrices,&ft1->matrices);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_send,&send_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_send,&submat);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_send,&array);CHKERRQ(ierr);
  if(n_send) {
    for (j=0, i=1; i<ft->n_neigh_lb; i++){
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(ft1->localG,isindex,NULL,MAT_INITIAL_MATRIX,&submat[j]);CHKERRQ(ierr);
      ierr = MatDenseGetArray(submat[j],&array[j]);CHKERRQ(ierr);   
      ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
      ierr = MPI_Isend(array[j],ft1->n_rbm*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&send_reqs[j]);CHKERRQ(ierr);
      ierr = ISDestroy(&isindex);CHKERRQ(ierr);
      j++;
    }
  }
  if(n_recv) {
    ierr = PetscMalloc1(n_recv,&recv_reqs);CHKERRQ(ierr);
    for (j=0,idx=0,i=1; i<ft->n_neigh_lb; i++){
      if (n_rbm_comm[ft->neigh_lb[i]]>0) {
	ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
	ierr = MPI_Irecv(&ft1->matrices[idx],n_rbm_comm[ft->neigh_lb[i]]*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&recv_reqs[j]);CHKERRQ(ierr);    
	idx += n_rbm_comm[ft->neigh_lb[i]]*ft->n_shared_lb[i]; 
	j++;
      }	  
    }
  }
  if(n_recv) { ierr = MPI_Waitall(n_recv,recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
  if(n_send) { ierr = MPI_Waitall(n_send,send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
  if(n_send) {
    for (i=0;i<n_send;i++) {
      ierr = MatDenseRestoreArray(submat[i],&array[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&submat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(array);CHKERRQ(ierr);
    ierr = PetscFree(submat);CHKERRQ(ierr);
  }

  /* store received matrices in Gholder */
  ft1->n_Gholder = n_recv;
  ierr = PetscMalloc1(n_recv,&ft1->Gholder);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_recv,&ft1->neigh_holder);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*n_recv,&ft1->neigh_holder[0]);CHKERRQ(ierr);
  for (i=1;i<n_recv;i++) { 
    ft1->neigh_holder[i] = ft1->neigh_holder[i-1] + 2;
  }
  for (i=0,idx=0,k=1; k<ft->n_neigh_lb; k++){
    if (n_rbm_comm[ft->neigh_lb[k]]>0) {
      ft1->neigh_holder[i][0] = ft->neigh_lb[k];
      ft1->neigh_holder[i][1] = k;
      ierr  = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_shared_lb[k],n_rbm_comm[ft->neigh_lb[k]],&ft1->matrices[idx],&ft1->Gholder[i++]);CHKERRQ(ierr);
      idx  += n_rbm_comm[ft->neigh_lb[k]]*ft->n_shared_lb[k];
    }
  }

  /** perfoming the actual multiplication G_{rankG}^T*G_{neigh_rankG>=rankG} */   
  if (ft1->n_rbm) {
    ierr = PetscMalloc1(ft->n_lambda_local,&idxm);CHKERRQ(ierr);
    ierr = PetscMalloc1(ft1->max_n_rbm,&idxn);CHKERRQ(ierr);
    for (i=0;i<ft->n_lambda_local;i++) idxm[i] = i;
    for (i=0;i<ft1->n_rbm;i++) idxn[i] = i;
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,localnnz,NULL,&aux_mat);CHKERRQ(ierr);
    ierr = MatZeroEntries(aux_mat);CHKERRQ(ierr);
    ierr = MatSetOption(aux_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatDenseGetArray(ft1->localG,&m_pointer);CHKERRQ(ierr);
    ierr = MatSetValuesBlocked(aux_mat,ft->n_lambda_local,idxm,ft1->n_rbm,idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(ft1->localG,&m_pointer);CHKERRQ(ierr);

    for (k=0; k<ft1->n_Gholder; k++){
      j   = ft1->neigh_holder[k][0];
      idx = ft1->neigh_holder[k][1];
      if (j>rankG) {
	for (k0=0;k0<n_rbm_comm[j];k0++) idxn[k0] = i++;
	ierr = MatDenseGetArray(ft1->Gholder[k],&m_pointer);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(aux_mat,ft->n_shared_lb[idx],ft->shared_lb[idx],n_rbm_comm[j],idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(ft1->Gholder[k],&m_pointer);CHKERRQ(ierr);	
      }
    }
    ierr = MatAssemblyBegin(aux_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(aux_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft1->n_rbm,localnnz,NULL,&result);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(ft1->localG,aux_mat,MAT_REUSE_MATRIX,PETSC_DEFAULT,&result);CHKERRQ(ierr);
    ierr = MatDestroy(&aux_mat);CHKERRQ(ierr);
    ierr = PetscFree(idxm);CHKERRQ(ierr);
    ierr = PetscFree(idxn);CHKERRQ(ierr);
    
    /* building structures for assembling the "global matrix" of the coarse problem */
    ierr   = PetscMalloc1(ft1->n_rbm,&idxm);CHKERRQ(ierr);
    ierr   = PetscMalloc1(localnnz,&idxn);CHKERRQ(ierr);
    /** row indices */
    idx = ft1->displ[rankG];
    for (i=0; i<ft1->n_rbm; i++) idxm[i] = i + idx;
    /** col indices */
    for (i=0; i<ft1->n_rbm; i++) idxn[i] = i + idx;
    for (j=1; j<ft->n_neigh_lb; j++) {
      k0  = n_rbm_comm[ft->neigh_lb[j]];
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
  total_c_coarse = nnz[0]*(n_rbm_comm[0]>0);
  c_count[0]     = nnz[0]*(n_rbm_comm[0]>0);
  c_displ[0]     = 0;
  idx            = n_rbm_comm[0];
  for (i=1;i<sizeG;i++) {
    total_c_coarse += nnz[idx]*(n_rbm_comm[i]>0);
    c_displ[i]      = c_displ[i-1] + c_count[i-1];
    c_count[i]      = nnz[idx]*(n_rbm_comm[i]>0);
    idx            += n_rbm_comm[i];
  }
  ierr = PetscMalloc1(total_c_coarse,&c_coarse);CHKERRQ(ierr);
  /* gather rows and columns*/
  ierr = MPI_Allgatherv(idxm,ft1->n_rbm,MPIU_INT,r_coarse,ft1->count_rbm,ft1->displ,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(idxn,localnnz,MPIU_INT,c_coarse,c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);

  /* gather values for the coarse problem's matrix and assemble it */ 
  for (k=0,k0=0,i=0; i<sizeG; i++) {
    if(n_rbm_comm[i]>0){
      ierr = PetscMPIIntCast(n_rbm_comm[i]*c_count[i],&i_mpi);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(i,&i_mpi1);CHKERRQ(ierr);
      if(i==rankG) {
	ierr = MPI_Bcast(m_pointer,i_mpi,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(ft1->coarse_problem,n_rbm_comm[i],&r_coarse[k],c_count[i],&c_coarse[k0],m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(result,&m_pointer);CHKERRQ(ierr);
	ierr = MatDestroy(&result);CHKERRQ(ierr);
      } else {
	ierr = PetscMalloc1(i_mpi,&m_pointer1);CHKERRQ(ierr);	  
	ierr = MPI_Bcast(m_pointer1,i_mpi,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(ft1->coarse_problem,n_rbm_comm[i],&r_coarse[k],c_count[i],&c_coarse[k0],m_pointer1,INSERT_VALUES);CHKERRQ(ierr);
	ierr = PetscFree(m_pointer1);CHKERRQ(ierr);
      }
      k0 += c_count[i];
      k  += n_rbm_comm[i];
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
  ierr = PetscFree(n_rbm_comm);CHKERRQ(ierr);
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
.  v_local  - the input vector (in this case v is a local copy of the global assembled vector)

   Output Parameter:
.  r_global  - the output vector. 

   Level: developer

.keywords: FETI1

.seealso: FETI1ApplyCoarseProblem_Private()
@*/
static PetscErrorCode FETI1ApplyCoarseProblem_Private(FETI ft,Vec v_local,Vec r_global)
{
  PetscErrorCode     ierr;
  FETI_1             *ft1 = (FETI_1*)ft->data;
  Vec                v_rbm; /* vec of dimension total_rbm */
  Vec                v0;    /* vec of dimension n_rbm */
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
  ierr = MatSolve(ft1->F_coarse,v_local,v_rbm);CHKERRQ(ierr);

  /* apply G: compute r = G*v_rbm */
  ierr = VecDuplicate(ft->lambda_local,&r_local);
  ierr = PetscMalloc1(ft1->max_n_rbm,&indices);CHKERRQ(ierr);
  /** mulplying by localG for the current processor */
  if(ft1->n_rbm) {
    for (i=0;i<ft1->n_rbm;i++) indices[i] = ft1->displ[rank] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft1->n_rbm,indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
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
    ierr = VecGetArrayRead(vec_holder,&m_pointer);CHKERRQ(ierr);
    ierr = MatMult(ft1->Gholder[j],v0,vec_holder);CHKERRQ(ierr);
    ierr = VecSetValues(r_local,ft->n_shared_lb[idx1],ft->shared_lb[idx1],m_pointer,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = ISDestroy(&subset);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec_holder,&m_pointer);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_holder);CHKERRQ(ierr); 
  }
  
  ierr = VecAssemblyBegin(r_local);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(r_local);CHKERRQ(ierr);

  ierr = VecScatterBegin(ft->l2g_lambda,r_local,r_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,r_local,r_global,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = VecDestroy(&v_rbm);CHKERRQ(ierr);
  ierr = VecDestroy(&r_local);CHKERRQ(ierr);
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
  if (ft1->n_rbm) { ierr = VecGetArrayRead(ft1->local_e,&sbuff);CHKERRQ(ierr);}   
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft1->n_rbm,MPIU_SCALAR,rbuff,ft1->count_rbm,ft1->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft1->n_rbm) { ierr = VecRestoreArrayRead(ft1->local_e,&sbuff);CHKERRQ(ierr);}

  ierr = FETI1ApplyCoarseProblem_Private(ft,asm_e,ft->lambda_global);CHKERRQ(ierr);
  if (ft1->n_rbm) { ierr = VecDestroy(&ft1->local_e);CHKERRQ(ierr);}
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
  FETI              ft   = (FETI)ft_ctx; 
  FETI_1            *ft1 = (FETI_1*)ft->data;
  Vec               asm_e;
  Vec               localv,y_local,y_local2;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  PetscInt          localsize;
  PetscMPIInt       rank;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(g_global,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  comm = PetscObjectComm((PetscObject)ft);
  if (y == g_global) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use g_global == y");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->total_rbm,&asm_e);CHKERRQ(ierr);
  ierr = VecScatterBegin(ft->l2g_lambda,g_global,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ft->l2g_lambda,g_global,ft->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (ft1->n_rbm) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft1->n_rbm,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft1->localG,ft->lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft1->n_rbm,MPIU_SCALAR,rbuff,ft1->count_rbm,ft1->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft1->n_rbm) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETI1ApplyCoarseProblem_Private(ft,asm_e,y);CHKERRQ(ierr);

  ierr = VecGetLocalSize(g_global,&localsize);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,localsize,&y_local);CHKERRQ(ierr);
  ierr = VecDuplicate(y_local,&y_local2);CHKERRQ(ierr);
  
  ierr = VecGetLocalVector(g_global,y_local2);CHKERRQ(ierr);

  ierr = VecGetLocalVector(y,y_local);CHKERRQ(ierr);
  
  ierr = VecAYPX(y_local,-1,y_local2);CHKERRQ(ierr);

  VecRestoreLocalVector(g_global,y_local2);
  VecRestoreLocalVector(y,y_local);
  
  //  VecView(y,PETSC_VIEWER_STDOUT_WORLD);
  /* VecSeqViewSynchronized(ft1->floatingComm,ft1->local_e); */
  /* VecSeqViewSynchronized(ft1->floatingComm,asm_e); */
  
  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  ierr = VecDestroy(&y_local);CHKERRQ(ierr);
  ierr = VecDestroy(&y_local2);CHKERRQ(ierr);
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
  ierr = PetscOptionsInsertString(mumps_options);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsInsertString(other_options);CHKERRQ(ierr);
  ierr = PetscOptionsInsert(argc,args,file);CHKERRQ(ierr);
    
  PetscFunctionReturn(0);
}
