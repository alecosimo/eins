#include <private/einssubdomain.h>
#include <petsc/private/petscimpl.h>


#undef __FUNCT__
#define __FUNCT__ "SubdomainDestroy"
/*@
  SubdomainDestroy - Destroy the allocated structures. 

  Level: developer

.keywords: FETI
.seealso: FETICreate()
@*/
PetscErrorCode  SubdomainDestroy(Subdomain *_sd)
{
  Subdomain      sd;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidPointer(_sd,1);
  sd = *_sd; *_sd = NULL;
  if (!sd) PetscFunctionReturn(0);
  if (--sd->refct > 0) PetscFunctionReturn(0);
  /* Free memory*/
  ierr = MatDestroy(&sd->localA);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->localRHS);CHKERRQ(ierr);
  ierr = ISDestroy(&sd->is_B_local);CHKERRQ(ierr);
  ierr = ISDestroy(&sd->is_I_local);CHKERRQ(ierr);
  ierr = ISDestroy(&sd->is_B_global);CHKERRQ(ierr);
  ierr = ISDestroy(&sd->is_I_global);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec1_N);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec2_N);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec1_D);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec1_B);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec2_B);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec1_global);CHKERRQ(ierr);
  ierr = MatDestroy(&sd->A_II);CHKERRQ(ierr);
  ierr = MatDestroy(&sd->A_BB);CHKERRQ(ierr);
  ierr = MatDestroy(&sd->A_IB);CHKERRQ(ierr);
  ierr = MatDestroy(&sd->A_BI);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sd->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sd->N_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sd->global_to_B);CHKERRQ(ierr);
  if (sd->n_neigh > -1) {
    ierr = ISLocalToGlobalMappingRestoreInfo(sd->mapping,&(sd->n_neigh),&(sd->neigh),&(sd->n_shared),&(sd->shared));CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&sd->mapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&sd->BtoNmap);CHKERRQ(ierr);
  if (sd->count) {ierr = PetscFree(sd->count);CHKERRQ(ierr);}
  if (sd->neighbours_set) {
    ierr = PetscFree(sd->neighbours_set[0]);CHKERRQ(ierr);
    ierr = PetscFree(sd->neighbours_set);CHKERRQ(ierr);
  }
  ierr = PetscFree(sd);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainCheckState"
/*@
  SubdomainCheckState - Check if every structure needed by FETI has been initialized.

  Level: developer

.keywords: FETI
.seealso: FETISetUp()
@*/
PetscErrorCode  SubdomainCheckState(Subdomain sd)
{
  PetscFunctionBegin;
  if (!sd->localA)      SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Local system matrix must be first defined");
  if (!sd->localRHS)    SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Local system RHS must be first defined");
  if (!sd->mapping)     SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Mapping from local to global DOF numbering must be first defined");
  if (!sd->vec1_global) SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: global working vector must be first created");
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SubdomainSetLocalRHS"
/*@
   SubdomainSetLocalRHS - Sets the local system RHS for the current process.

   Input Parameter:
.  sd  - The Subdomain context
.  rhs - The local system rhs

   Level: developer

.keywords: Subdomain, local system rhs

.seealso: SubdomainSetLocalMat(), SubdomainSetMapping()
@*/
PetscErrorCode SubdomainSetLocalRHS(Subdomain sd,Vec rhs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  /* this rutine is called from outside with a valid rhs*/
  ierr = VecDestroy(&sd->localRHS);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)rhs);CHKERRQ(ierr);
  sd->localRHS = rhs;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SubdomainCreateGlobalWorkingVec"
/*@
   SubdomainCreateGlobalWorkingVec - Creates the global (distributed) working vector by duplicating a given vector.

   Input Parameter:
.  sd  - The Subdomain context
.  vec - The global vector to use in the duplication

   Level: developer

.keywords: working global vector

@*/
PetscErrorCode SubdomainCreateGlobalWorkingVec(Subdomain sd,Vec vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  /* this rutine is called from outside with a valid vec*/
  ierr = VecDestroy(&sd->vec1_global);CHKERRQ(ierr);
  ierr = VecDuplicate(vec,&sd->vec1_global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SubdomainSetLocalMat"
/*@
   SubdomainSetLocalMat - Sets the local system matrix for the current process.

   Input Parameter:
.  sd    - The Subdomain context
.  local_mat - The local system matrix

   Level: beginner

.keywords: Subdomain, local system matrix

.seealso: SubdomainSetLocalRHS(), SubdomainSetMapping()
@*/
PetscErrorCode SubdomainSetLocalMat(Subdomain sd,Mat local_mat)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  /* this rutine is called from outside with a valid local_mat*/
  ierr = MatDestroy(&sd->localA);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)local_mat);CHKERRQ(ierr);
  sd->localA = local_mat;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SubdomainSetMapping"
/*@
   SubdomainSetMapping - Sets the mapping from local to global numbering of DOFs

   Input Parameter:
.  sd    - The Subdomain context
.  isg2l - A mapping from local to global numering of DOFs

   Level: developer

.keywords: Subdomain, local to global numbering of DOFs

.seealso: SubdomainSetLocalRHS(), SubdomainSetLocalMat()
@*/
PetscErrorCode SubdomainSetMapping(Subdomain sd,ISLocalToGlobalMapping isg2l)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  /* this rutine is called from outside with a valid isg2l*/
  ierr = ISLocalToGlobalMappingDestroy(&sd->mapping);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)isg2l);CHKERRQ(ierr);
  sd->mapping = isg2l;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainComputeSubmatrices"
/*@
  SubdomainComputeSubmatrices - Compute the subsmatrices A_II, A_IB, A_BI, A_BB of the local system matrix.

   Input Parameter:
.  sd      - The Subdomain context
.  reuse   - Either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
.  onlyAbb - Compute only A_BB

  Notes: The first time this is called you should use reuse equal to
  MAT_INITIAL_MATRIX. Any additional calls to this routine with a
  local mat of the same nonzero structure and with a call of
  MAT_REUSE_MATRIX will reuse the matrix generated the first time.

  Level: developer

.keywords: FETI
.seealso: FETICreate()
@*/
PetscErrorCode SubdomainComputeSubmatrices(Subdomain sd, MatReuse reuse, PetscBool onlyAbb)
{
  PetscErrorCode ierr;
  PetscBool      issbaij;

  PetscFunctionBegin; 
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDestroy(&sd->A_II);CHKERRQ(ierr);
    ierr = MatDestroy(&sd->A_IB);CHKERRQ(ierr);
    ierr = MatDestroy(&sd->A_BI);CHKERRQ(ierr);
    ierr = MatDestroy(&sd->A_BB);CHKERRQ(ierr);
  }

  ierr = MatGetSubMatrix(sd->localA,sd->is_B_local,sd->is_B_local,reuse,&sd->A_BB);CHKERRQ(ierr);
  if(!onlyAbb){
    ierr = MatGetSubMatrix(sd->localA,sd->is_I_local,sd->is_I_local,reuse,&sd->A_II);CHKERRQ(ierr); 
    ierr = PetscObjectTypeCompare((PetscObject)sd->localA,MATSEQSBAIJ,&issbaij);CHKERRQ(ierr);
    if (!issbaij) {
      ierr = MatGetSubMatrix(sd->localA,sd->is_I_local,sd->is_B_local,reuse,&sd->A_IB);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(sd->localA,sd->is_B_local,sd->is_I_local,reuse,&sd->A_BI);CHKERRQ(ierr);
    } else {
      Mat newmat;
      ierr = MatConvert(sd->localA,MATSEQBAIJ,MAT_INITIAL_MATRIX,&newmat);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(newmat,sd->is_I_local,sd->is_B_local,reuse,&sd->A_IB);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(newmat,sd->is_B_local,sd->is_I_local,reuse,&sd->A_BI);CHKERRQ(ierr);
      ierr = MatDestroy(&newmat);CHKERRQ(ierr);
    }
    ierr = MatSetOption(sd->A_II,MAT_SYMMETRIC,issbaij);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainCreate"
/*@
  SubdomainCreate - Creates the basic structures for dealing with subdomain information, such
  as the local system matrix, the local rhs and the mapping from local to global numering of
  DOFs.

   Input Parameter:
.  comm - The MPI communicator

   Output Parameter:
.  sd    - Pointer to the Subdomain context

  Level: developer

.keywords: FETI
.seealso: FETICreate()
@*/
PetscErrorCode  SubdomainCreate(MPI_Comm comm, Subdomain *_sd)
{
  Subdomain      sd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_sd,1);
  ierr = PetscCalloc1(1,&sd);CHKERRQ(ierr);
  *_sd = sd; sd->refct = 1;
  sd->comm             = comm;
  sd->count            = 0;
  sd->neighbours_set   = 0;
  sd->is_B_local       = 0;
  sd->is_I_local       = 0;
  sd->is_B_global      = 0;
  sd->is_I_global      = 0;
  sd->vec1_N           = 0;
  sd->vec2_N           = 0;
  sd->vec1_D           = 0;
  sd->vec1_B           = 0;
  sd->vec2_B           = 0;
  sd->vec1_global      = 0;
  sd->global_to_D      = 0;
  sd->N_to_B           = 0;
  sd->global_to_B      = 0;
  sd->mapping          = 0;
  sd->BtoNmap          = 0;
  sd->n_neigh          = -1;  
  PetscFunctionReturn(0);  
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainSetUp"
/*@
  SubdomainSetUp - Setups subdomain infromartion by mainly it computes VecScatters information. 

   Input Parameter:
.  sd    - The Subdomain context
.  fetisetupcalled - It takes the value of feti->setupcalled

  Level: developer

.keywords: FETI
.seealso: FETISetUp()
@*/
PetscErrorCode SubdomainSetUp(Subdomain sd, PetscBool fetisetupcalled)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* first time creation, get info on substructuring */
  if (!fetisetupcalled) {
    PetscInt    n_I;
    PetscInt    *idx_I_local,*idx_B_local,*idx_I_global,*idx_B_global;
    PetscInt    *array;
    PetscInt    i,j,k,rank;

    ierr = MPI_Comm_rank(sd->comm,&rank);CHKERRQ(ierr);
    /* get info on mapping */
    ierr = ISLocalToGlobalMappingGetSize(sd->mapping,&sd->n);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetInfo(sd->mapping,&(sd->n_neigh),&(sd->neigh),&(sd->n_shared),&(sd->shared));CHKERRQ(ierr);
    /* Identifying interior and interface nodes, in local numbering */
    ierr = PetscMalloc1(sd->n,&array);CHKERRQ(ierr);
    ierr = PetscMemzero(array,sd->n*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0;i<sd->n_neigh;i++)
      for (j=0;j<sd->n_shared[i];j++)
	array[sd->shared[i][j]] += 1;
    /* Creating local and global index sets for interior and inteface nodes. */
    ierr = PetscMalloc1(sd->n,&idx_I_local);CHKERRQ(ierr);
    ierr = PetscMalloc1(sd->n,&idx_B_local);CHKERRQ(ierr);
    for (i=0, sd->n_B=0, n_I=0; i<sd->n; i++) {
      if (!array[i]) {
        idx_I_local[n_I] = i;
        n_I++;
      } else {
        idx_B_local[sd->n_B] = i;
	array[i] = sd->n_B;
        sd->n_B++;
      }
    }
    ierr = PetscCalloc1((size_t)sd->n_B,&sd->count);CHKERRQ(ierr);
    /* Count total number of neigh per node */
    k=0;
    for (i=1;i<sd->n_neigh;i++) {//not count myself
      k += sd->n_shared[i];
      for (j=0;j<sd->n_shared[i];j++) {
	sd->count[array[sd->shared[i][j]]] += 1;
      }
    }
    /* Allocate space for storing the set of neighbours for each node */
    ierr = PetscMalloc1(sd->n_B,&sd->neighbours_set);CHKERRQ(ierr);
    ierr = PetscMalloc1(k,&sd->neighbours_set[0]);CHKERRQ(ierr);
    for (i=1;i<sd->n_B;i++) { 
      sd->neighbours_set[i] = sd->neighbours_set[i-1]+sd->count[i-1];
    }
    /* Get information for sharing subdomains */
    ierr = PetscMemzero(sd->count,sd->n_B*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=1;i<sd->n_neigh;i++) {//not count myself
      for (j=0;j<sd->n_shared[i];j++) {
	k = array[sd->shared[i][j]];
	sd->neighbours_set[k][sd->count[k]++] = sd->neigh[i];
      }
    }
    /* sort set of sharing subdomains */
    for (i=0;i<sd->n_B;i++) {
      ierr = PetscSortRemoveDupsInt(&sd->count[i],sd->neighbours_set[i]);CHKERRQ(ierr);
    }
    
    /* Getting the global numbering */
    idx_B_global = idx_I_local + n_I; /* Just avoiding allocating extra memory, since we have vacant space */
    idx_I_global = idx_B_local + sd->n_B;
    ierr         = ISLocalToGlobalMappingApply(sd->mapping,sd->n_B,idx_B_local,idx_B_global);CHKERRQ(ierr);
    ierr         = ISLocalToGlobalMappingApply(sd->mapping,n_I,idx_I_local,idx_I_global);CHKERRQ(ierr);
    /* Creating the index sets */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sd->n_B,idx_B_local,PETSC_COPY_VALUES, &sd->is_B_local);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,sd->n_B,idx_B_global,PETSC_COPY_VALUES,&sd->is_B_global);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_local,PETSC_COPY_VALUES, &sd->is_I_local);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_I,idx_I_global,PETSC_COPY_VALUES,&sd->is_I_global);CHKERRQ(ierr);
    /* Freeing memory */
    ierr = PetscFree(idx_B_local);CHKERRQ(ierr);
    ierr = PetscFree(idx_I_local);CHKERRQ(ierr);
    ierr = PetscFree(array);CHKERRQ(ierr);

    /* Creating work vectors vec1_N, vec1_B and vec1_D */
    ierr = VecDuplicate(sd->localRHS,&sd->vec1_N);CHKERRQ(ierr);
    ierr = VecDuplicate(sd->localRHS,&sd->vec2_N);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,sd->n_B,&sd->vec1_B);CHKERRQ(ierr);
    ierr = VecDuplicate(sd->vec1_B,&sd->vec2_B);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,sd->n-sd->n_B,&sd->vec1_D);CHKERRQ(ierr);
    /* Creating the scatter contexts */
    ierr = VecScatterCreate(sd->vec1_N,sd->is_B_local,sd->vec1_B,(IS)0,&sd->N_to_B);CHKERRQ(ierr);
    ierr = VecScatterCreate(sd->vec1_global,sd->is_I_global,sd->vec1_D,(IS)0,&sd->global_to_D);CHKERRQ(ierr);
    ierr = VecScatterCreate(sd->vec1_global,sd->is_B_global,sd->vec1_B,(IS)0,&sd->global_to_B);CHKERRQ(ierr);
    /* map from boundary to local */
    ierr = ISLocalToGlobalMappingCreateIS(sd->is_B_local,&sd->BtoNmap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



