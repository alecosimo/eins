#include <../src/feti/einsfeti1.h>
#include <einssys.h>

/* private functions*/
static PetscErrorCode FETI1BuildLambdaAndB_Private(FETI);
static PetscErrorCode FETI1SetUpNeumannSolver_Private(FETI);
static PetscErrorCode FETI1ComputeMatrixG_Private(FETI);
static PetscErrorCode FETI1BuildInterfaceProblem_Private(FETI);

#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI1"
/*@
   FETIDestroy_FETI1 - Destroys the FETI-1 context

   Input Parameters:
.  ft - the FETI context

.seealso FETICreate_FETI1
@*/
PetscErrorCode FETIDestroy_FETI1(FETI ft);
PetscErrorCode FETIDestroy_FETI1(FETI ft)
{
  PetscErrorCode ierr;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  ierr = MatDestroy(&ft1->localG);CHKERRQ(ierr);
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
PetscErrorCode FETISetUp_FETI1(FETI ft);
PetscErrorCode FETISetUp_FETI1(FETI ft)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FETI1BuildLambdaAndB_Private(ft);CHKERRQ(ierr);
  ierr = FETI1SetUpNeumannSolver_Private(ft);CHKERRQ(ierr);  
  ierr = FETI1ComputeMatrixG_Private(ft);CHKERRQ(ierr);
  ierr = FETI1BuildInterfaceProblem_Private(ft);CHKERRQ(ierr);
  ierr = FETIBuildInterfaceKSP(ft);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "FETICreate_FETI1"
/*@
   FETI1 - Implementation of the FETI-1 method. Some comments about options can be put here!

   Options:
.  -feti_fullyredundant: use fully redundant Lagrange multipliers.
.  -feti_interface_<ksp or pc option>: options for the KSP for the interface problem
.  -feti1_neumann_<ksp or pc option>: for setting pc and ksp options for the neumann solver. 
    
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

  feti1->localG                = 0;  
  /* function pointers */
  ft->ops->setup               = FETISetUp_FETI1;
  ft->ops->destroy             = FETIDestroy_FETI1;
  ft->ops->setfromoptions      = 0;//FETISetFromOptions_FETI1;
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
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Vec            lambda_global;
  IS             IS_l2g_lambda;
  IS             subset,subset_mult,subset_n;
  PetscBool      fully_redundant;
  PetscInt       i,j,s,n_boundary_dofs,n_global_lambda,partial_sum;
  PetscInt       cum,n_local_lambda,n_lambda_for_dof,dual_size,n_neg_values,n_pos_values;
  PetscMPIInt    rank;
  PetscInt       *dual_dofs_boundary_indices,*aux_local_numbering_1;
  const PetscInt *aux_global_numbering,*indices;
  PetscInt       *aux_sums,*cols_B_delta,*l2g_indices;
  PetscScalar    *array,*vals_B_delta;
  PetscInt       *aux_local_numbering_2;
  PetscScalar    scalar_value;
  Subdomain      sd = ft->subdomain;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Default type of lagrange multipliers is non-redundant */
  fully_redundant = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-feti_fullyredundant",&fully_redundant,NULL);CHKERRQ(ierr);

  /* Evaluate local and global number of lagrange multipliers */
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  n_local_lambda = 0;
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
    n_local_lambda += j;
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
    SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in %s: global number of multipliers mismatch! (%d!=%d)\n",__FUNCT__,ft->n_lambda,i);
  }
  ft->n_local_lambda = n_local_lambda;
  /* Compute B_delta (local actions) */
  ierr = PetscMalloc1(sd->n_neigh,&aux_sums);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&l2g_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&vals_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_lambda,&cols_B_delta);CHKERRQ(ierr);
  ierr = ISGetIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
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
        l2g_indices    [partial_sum+s]=aux_sums[s]+n_neg_values-s-1+n_global_lambda;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=-1.0;
      }
      for (s=0;s<n_pos_values;s++) {
        l2g_indices    [partial_sum+s+n_neg_values]=aux_sums[n_neg_values]+s+n_global_lambda;
        cols_B_delta   [partial_sum+s+n_neg_values]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s+n_neg_values]=1.0;
      }
      partial_sum += j;
    } else {
      /* l2g_indices and default cols and vals of B_delta */
      for (s=0;s<j;s++) {
        l2g_indices    [partial_sum+s]=n_global_lambda+s;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=0.0;
      }
      /* B_delta */
      if ( n_neg_values > 0 ) { /* there's a rank next to me to the left */
        vals_B_delta   [partial_sum+n_neg_values-1]=-1.0;
      }
      if ( n_neg_values < j ) { /* there's a rank next to me to the right */
        vals_B_delta   [partial_sum+n_neg_values]=1.0;
      }
      partial_sum += j;
    }
    cum += aux_local_numbering_2[i];
  }
  ierr = ISRestoreIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_mult);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = PetscFree(aux_sums);CHKERRQ(ierr);
  ierr = PetscFree(dual_dofs_boundary_indices);CHKERRQ(ierr);

  /* Local to global mapping for lagrange multipliers */
  ierr = VecCreate(PETSC_COMM_SELF,&ft->lambda_local);CHKERRQ(ierr);
  ierr = VecSetSizes(ft->lambda_local,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
  ierr = VecSetType(ft->lambda_local,VECSEQ);CHKERRQ(ierr);
  ierr = VecCreate(comm,&lambda_global);CHKERRQ(ierr);
  ierr = VecSetSizes(lambda_global,PETSC_DECIDE,ft->n_lambda);CHKERRQ(ierr);
  ierr = VecSetType(lambda_global,VECMPI);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_OWN_POINTER,&IS_l2g_lambda);CHKERRQ(ierr);
  ierr = VecScatterCreate(ft->lambda_local,(IS)0,lambda_global,IS_l2g_lambda,&ft->l2g_lambda);CHKERRQ(ierr);
  ierr = ISDestroy(&IS_l2g_lambda);CHKERRQ(ierr);

  /* Create local part of B_delta */
  ierr = MatCreate(PETSC_COMM_SELF,&ft->B_delta);CHKERRQ(ierr);
  ierr = MatSetSizes(ft->B_delta,n_local_lambda,sd->n_B,n_local_lambda,sd->n_B);CHKERRQ(ierr);
  ierr = MatSetType(ft->B_delta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(ft->B_delta,1,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(ft->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<n_local_lambda;i++) {
    ierr = MatSetValue(ft->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(ft->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (ft->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(vals_B_delta);CHKERRQ(ierr);
  ierr = PetscFree(cols_B_delta);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda_global);CHKERRQ(ierr);

  MatSeqViewSynchronized(ft->B_delta);
   
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1DestroyMatF"
PetscErrorCode FETI1DestroyMatF(Mat A);
PetscErrorCode FETI1DestroyMatF(Mat A)
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
#define __FUNCT__ "FETI1MatMult"
PetscErrorCode FETI1MatMult(Mat F, Vec lambda_global, Vec y); /* y=F*lambda_global */
PetscErrorCode FETI1MatMult(Mat F, Vec lambda_global, Vec y)
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
  ierr = FETICreateFMat(ft,(void (*)(void))FETI1MatMult,(void (*)(void))FETI1DestroyMatF);CHKERRQ(ierr);
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
#define __FUNCT__ "FETI1ComputeMatrixG_Private"
/*@
   FETI1ComputeRBM_Private - Computes the Rigid Body Modes and the application of the operator B to them.

   Input Parameters:
.  ft - the FETI context

@*/
static PetscErrorCode FETI1ComputeMatrixG_Private(FETI ft)
{
  PetscErrorCode ierr;
  Subdomain      sd = ft->subdomain;
  PetscInt       infog,rank;
  MPI_Comm       comm;
  Mat            rbm,x; 
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  ierr   = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ft1->localG = 0;
  /* get number of rigid body modes */
  ierr   = MatMumpsGetInfog(ft1->F_neumann,28,&infog);CHKERRQ(ierr);
  if(infog){
    /* Compute rigid body modes */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,infog,NULL,&rbm);CHKERRQ(ierr);
    ierr = MatDuplicate(rbm,MAT_DO_NOT_COPY_VALUES,&x);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(rbm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMumpsSetIcntl(ft1->F_neumann,25,-1);CHKERRQ(ierr);
    ierr = MatMatSolve(ft1->F_neumann,x,rbm);CHKERRQ(ierr);
    ierr = MatDestroy(&x);CHKERRQ(ierr);
    
    ierr = MatGetSubMatrix(rbm,sd->is_B_local,NULL,MAT_INITIAL_MATRIX,&x);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_local_lambda,infog,NULL,&ft1->localG);CHKERRQ(ierr);
    ierr = MatMatMult(ft->B_delta,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&ft1->localG);CHKERRQ(ierr);    
    if(rank==1){
      PetscPrintf(PETSC_COMM_SELF,"\n==================================================\n");
      PetscPrintf(PETSC_COMM_SELF,"Printing rigid body modes\n");
      PetscPrintf(PETSC_COMM_SELF,"==================================================\n");

      MatView(ft1->localG,PETSC_VIEWER_STDOUT_SELF);
    }
    ierr = MatDestroy(&x);CHKERRQ(ierr);
    ierr = MatDestroy(&rbm);CHKERRQ(ierr);
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
    SETERRQ(PETSC_COMM_WORLD,1,"EINS only supports MUMPS for the solution of the Neumann problem");
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
    
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc,&ft1->F_neumann);CHKERRQ(ierr);
    /* sequential ordering */
    ierr = MatMumpsSetIcntl(ft1->F_neumann,7,2);CHKERRQ(ierr);
    /* Null row pivot detection */
    ierr = MatMumpsSetIcntl(ft1->F_neumann,24,1);CHKERRQ(ierr);
    /* threshhold for row pivot detection */
    ierr = MatMumpsSetCntl(ft1->F_neumann,3,1.e-6);CHKERRQ(ierr);

    /* Maybe the following two options should be given as external options and not here*/
    ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCFactorSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ft->ksp_neumann);CHKERRQ(ierr);
  } else {
    ierr = KSPSetOperators(ft->ksp_neumann,sd->localA,sd->localA);CHKERRQ(ierr);
  }
  /* Set Up KSP for Neumann problem: here the factorization takes place!!! */
  ierr = KSPSetUp(ft->ksp_neumann);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
