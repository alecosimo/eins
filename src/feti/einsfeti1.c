#include <../src/feti/einsfeti1.h>
#include <einssys.h>


static PetscErrorCode FETI1BuildLambdaAndB_Private(FETI);
  
#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI1"
/*
   FETIDestroy_FETI1 - Destroys the FETI-1 context

   Input Parameters:
.  ft - the FETI context

.seealso FETICreate_FETI1
 */
PetscErrorCode FETIDestroy_FETI1(FETI ft)
{
  FETI_1*        feti1 = (FETI_1*)ft->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETI1"
/*
   FETISetUp_FETI1 - Prepares the structures needed by the FETI-1 solver.

   Input Parameters:
.  ft - the FETI context

*/
PetscErrorCode FETISetUp_FETI1(FETI ft)
{
  PetscErrorCode ierr;
  FETI_1*        feti1 = (FETI_1*)ft->data;

  PetscFunctionBegin;
  FETI1BuildLambdaAndB_Private(ft);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICreate_FETI1"
/*@
   FETI1 - Implementation of the FETI-1 method. Some comments about options can be put here!

   Options:
.  feti_fullyredundant: use fully redundant Lagrange multipliers.
    
   Level: beginner

.keywords: FETI, FETI-1
@*/
PETSC_EXTERN PetscErrorCode FETICreate_FETI1(FETI ft)
{
  PetscErrorCode      ierr;
  FETI_1*             feti1;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ft,&feti1);CHKERRQ(ierr);
  ft->data  = (void*)feti1;

  /* function pointers */
  ft->ops->setup               = FETISetUp_FETI1;
  ft->ops->destroy             = FETIDestroy_FETI1;
  ft->ops->setfromoptions      = 0;//FETISetFromOptions_FETI1;
  ft->ops->view                = 0;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETI1BuildLambdaAndB_Private"
/*
   FETI1BuildLambdaAndB_Private - Computes the B operator and the vector lambda of 
   the interface problem.

   Input Parameters:
.  ft - the FETI context

 */
static PetscErrorCode FETI1BuildLambdaAndB_Private(FETI ft)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Vec            lambda_global;
  IS             IS_l2g_lambda;
  IS             subset,subset_mult,subset_n;
  PetscBool      skip_node,fully_redundant;
  PetscInt       i,j,k,s,n_boundary_dofs,n_global_lambda,partial_sum;
  PetscInt       cum,n_local_lambda,n_lambda_for_dof,dual_size,n_neg_values,n_pos_values;
  PetscMPIInt    rank,size,buf_size,neigh;
  PetscScalar    scalar_value;
  PetscInt       *dual_dofs_boundary_indices,*aux_local_numbering_1;
  const PetscInt *aux_global_numbering,*indices;
  PetscInt       *aux_sums,*cols_B_delta,*l2g_indices;
  PetscScalar    *array,*scaling_factors,*vals_B_delta;
  PetscInt       *aux_local_numbering_2;
  /* For communication of scaling factors */
  PetscInt       *ptrs_buffer,neigh_position;
  PetscScalar    **all_factors,*send_buffer,*recv_buffer;
  MPI_Request    *send_reqs,*recv_reqs;
  Subdomain      sd = ft->subdomain;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

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
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_2,PETSC_OWN_POINTER,&subset_mult);CHKERRQ(ierr);
  ierr = ISSubsetNumbering(subset,subset_mult,&i,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);
  if (i != ft->n_lambda) {
    SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in %s: global number of multipliers mismatch! (%d!=%d)\n",__FUNCT__,ft->n_lambda,i);
  }

  
  #if 0
  /* /\* init data for scaling factors exchange *\/ */
  /* partial_sum = 0; */
  /* j = 0; */
  /* ierr = PetscMalloc1(->n_neigh,&ptrs_buffer);CHKERRQ(ierr); */
  /* ierr = PetscMalloc1(pcis->n_neigh-1,&send_reqs);CHKERRQ(ierr); */
  /* ierr = PetscMalloc1(pcis->n_neigh-1,&recv_reqs);CHKERRQ(ierr); */
  /* ierr = PetscMalloc1(pcis->n,&all_factors);CHKERRQ(ierr); */
  /* ptrs_buffer[0]=0; */
  /* for (i=1;i<pcis->n_neigh;i++) { */
  /*   partial_sum += pcis->n_shared[i]; */
  /*   ptrs_buffer[i] = ptrs_buffer[i-1]+pcis->n_shared[i]; */
  /* } */
  /* ierr = PetscMalloc1(partial_sum,&send_buffer);CHKERRQ(ierr); */
  /* ierr = PetscMalloc1(partial_sum,&recv_buffer);CHKERRQ(ierr); */
  /* ierr = PetscMalloc1(partial_sum,&all_factors[0]);CHKERRQ(ierr); */
  /* for (i=0;i<pcis->n-1;i++) { */
  /*   j = mat_graph->count[i]; */
  /*   all_factors[i+1]=all_factors[i]+j; */
  /* } */
  /* /\* scatter B scaling to N vec *\/ */
  /* ierr = VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr); */
  /* ierr = VecScatterEnd(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr); */
  /* /\* communications *\/ */
  /* ierr = VecGetArrayRead(pcis->vec1_N,(const PetscScalar**)&array);CHKERRQ(ierr); */
  /* for (i=1;i<pcis->n_neigh;i++) { */
  /*   for (j=0;j<pcis->n_shared[i];j++) { */
  /*     send_buffer[ptrs_buffer[i-1]+j]=array[pcis->shared[i][j]]; */
  /*   } */
  /*   ierr = PetscMPIIntCast(ptrs_buffer[i]-ptrs_buffer[i-1],&buf_size);CHKERRQ(ierr); */
  /*   ierr = PetscMPIIntCast(pcis->neigh[i],&neigh);CHKERRQ(ierr); */
  /*   ierr = MPI_Isend(&send_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&send_reqs[i-1]);CHKERRQ(ierr); */
  /*   ierr = MPI_Irecv(&recv_buffer[ptrs_buffer[i-1]],buf_size,MPIU_SCALAR,neigh,0,comm,&recv_reqs[i-1]);CHKERRQ(ierr); */
  /* } */
  /* ierr = VecRestoreArrayRead(pcis->vec1_N,(const PetscScalar**)&array);CHKERRQ(ierr); */
  /* ierr = MPI_Waitall((pcis->n_neigh-1),recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr); */
  /* /\* put values in correct places *\/ */
  /* for (i=1;i<pcis->n_neigh;i++) { */
  /*   for (j=0;j<pcis->n_shared[i];j++) { */
  /*     k = pcis->shared[i][j]; */
  /*     neigh_position = 0; */
  /*     while(mat_graph->neighbours_set[k][neigh_position] != pcis->neigh[i]) {neigh_position++;} */
  /*     all_factors[k][neigh_position]=recv_buffer[ptrs_buffer[i-1]+j]; */
  /*   } */
  /* } */
  /* ierr = MPI_Waitall((pcis->n_neigh-1),send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr); */
  /* ierr = PetscFree(send_reqs);CHKERRQ(ierr); */
  /* ierr = PetscFree(recv_reqs);CHKERRQ(ierr); */
  /* ierr = PetscFree(send_buffer);CHKERRQ(ierr); */
  /* ierr = PetscFree(recv_buffer);CHKERRQ(ierr); */
  /* ierr = PetscFree(ptrs_buffer);CHKERRQ(ierr); */

  /* Compute B and B_delta (local actions) */
  ierr = PetscMalloc1(pcis->n_neigh,&aux_sums);CHKERRQ(ierr);
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
    
    while(n_neg_values < j && mat_graph->neighbours_set[aux_local_numbering_1[i]][n_neg_values] < rank){
      n_neg_values++;
    }
    
    n_pos_values = j - n_neg_values;
    if (fully_redundant) {
      for (s=0;s<n_neg_values;s++) {
        l2g_indices    [partial_sum+s]=aux_sums[s]+n_neg_values-s-1+n_global_lambda;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=-1.0;
        scaling_factors[partial_sum+s]=array[s];
      }
      for (s=0;s<n_pos_values;s++) {
        l2g_indices    [partial_sum+s+n_neg_values]=aux_sums[n_neg_values]+s+n_global_lambda;
        cols_B_delta   [partial_sum+s+n_neg_values]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s+n_neg_values]=1.0;
        scaling_factors[partial_sum+s+n_neg_values]=array[s+n_neg_values];
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
  ierr = PetscFree(aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscFree(dual_dofs_boundary_indices);CHKERRQ(ierr);
  ierr = PetscFree(all_factors[0]);CHKERRQ(ierr);
  ierr = PetscFree(all_factors);CHKERRQ(ierr);

  /* Local to global mapping of fetidpmat */
  ierr = VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSetSizes(fetidpmat_ctx->lambda_local,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
  ierr = VecSetType(fetidpmat_ctx->lambda_local,VECSEQ);CHKERRQ(ierr);
  ierr = VecCreate(comm,&lambda_global);CHKERRQ(ierr);
  ierr = VecSetSizes(lambda_global,PETSC_DECIDE,fetidpmat_ctx->n_lambda);CHKERRQ(ierr);
  ierr = VecSetType(lambda_global,VECMPI);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_OWN_POINTER,&IS_l2g_lambda);CHKERRQ(ierr);
  ierr = VecScatterCreate(fetidpmat_ctx->lambda_local,(IS)0,lambda_global,IS_l2g_lambda,&fetidpmat_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = ISDestroy(&IS_l2g_lambda);CHKERRQ(ierr);

  /* Create local part of B_delta */
  ierr = MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_delta);CHKERRQ(ierr);
  ierr = MatSetSizes(fetidpmat_ctx->B_delta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B);CHKERRQ(ierr);
  ierr = MatSetType(fetidpmat_ctx->B_delta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(fetidpmat_ctx->B_delta,1,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(fetidpmat_ctx->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<n_local_lambda;i++) {
    ierr = MatSetValue(fetidpmat_ctx->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(vals_B_delta);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (fully_redundant) {
    ierr = MatCreate(PETSC_COMM_SELF,&ScalingMat);CHKERRQ(ierr);
    ierr = MatSetSizes(ScalingMat,n_local_lambda,n_local_lambda,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
    ierr = MatSetType(ScalingMat,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(ScalingMat,1,NULL);CHKERRQ(ierr);
    for (i=0;i<n_local_lambda;i++) {
      ierr = MatSetValue(ScalingMat,i,i,scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMatMult(ScalingMat,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);
    ierr = MatDestroy(&ScalingMat);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);
    ierr = MatSetSizes(fetidpmat_ctx->B_Ddelta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B);CHKERRQ(ierr);
    ierr = MatSetType(fetidpmat_ctx->B_Ddelta,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(fetidpmat_ctx->B_Ddelta,1,NULL);CHKERRQ(ierr);
    for (i=0;i<n_local_lambda;i++) {
      ierr = MatSetValue(fetidpmat_ctx->B_Ddelta,i,cols_B_delta[i],scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscFree(scaling_factors);CHKERRQ(ierr);
  ierr = PetscFree(cols_B_delta);CHKERRQ(ierr);

  /* Create some vectors needed by fetidp */
  ierr = VecDuplicate(pcis->vec1_B,&fetidpmat_ctx->temp_solution_B);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&fetidpmat_ctx->temp_solution_D);CHKERRQ(ierr);

  test_fetidp = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-fetidp_check",&test_fetidp,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&lambda_global);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
