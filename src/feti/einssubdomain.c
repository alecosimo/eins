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
  ierr = VecDestroy(&sd->vec1_D);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec1_B);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec1_global);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sd->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sd->N_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sd->global_to_B);CHKERRQ(ierr);
  if (sd->n_neigh > -1) {
    ierr = ISLocalToGlobalMappingRestoreInfo(sd->mapping,&(sd->n_neigh),&(sd->neigh),&(sd->n_shared),&(sd->shared));CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&sd->mapping);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&sd->BtoNmap);CHKERRQ(ierr);
  
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
  if (!sd->localA)   SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Local system matrix must be first defined");
  if (!sd->localRHS) SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Local system RHS must be first defined");
  if (!sd->mapping)  SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Mapping from local to global DOF numbering must be first defined");
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
#define __FUNCT__ "SubdomainCreate"
/*@
  SubdomainCreate - Creates the basic structures for dealing with subdomain information, such
  as the local system matrix, the local rhs and the mapping from local to global numering of
  DOFs.

  Level: developer

.keywords: FETI
.seealso: FETICreate()
@*/
PetscErrorCode  SubdomainCreate(Subdomain *_sd)
{
  Subdomain      sd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_sd,1);
  ierr = PetscCalloc1(1,&sd);CHKERRQ(ierr);
  *_sd = sd; sd->refct = 1;
  sd->is_B_local       = 0;
  sd->is_I_local       = 0;
  sd->is_B_global      = 0;
  sd->is_I_global      = 0;
  sd->vec1_N           = 0;
  sd->vec1_D           = 0;
  sd->vec1_B           = 0;
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
    PetscInt    i,j;

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
        sd->n_B++;
      }
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

    /* Creating work vectors vec1_N and vec1_B */
    ierr = VecDuplicate(sd->localRHS,&sd->vec1_N);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,sd->n_B,&sd->vec1_B);CHKERRQ(ierr);
    /* Creating the scatter contexts */
    ierr = VecScatterCreate(sd->vec1_N,sd->is_B_local,sd->vec1_B,(IS)0,&sd->N_to_B);CHKERRQ(ierr);
    /* map from boundary to local */
    ierr = ISLocalToGlobalMappingCreateIS(sd->is_B_local,&sd->BtoNmap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainCreateVecScatter"
/*@ SubdomainCreateVecScatters - It createss the VecScatters
  global_to_D and global_to_B. It will also create the vector vec1_D
  for the scatter global_to_D. 

   Input Parameter:
.  sd    - The Subdomain context.
.  _vec1_global - A reference to the global vector (created by the user).

  Level: developer

  Notes: 
     When calling this function, pre-existing instancies of
     VecScatters global_to_D and global_to_B will be destroyed. The
     same comment applies for the vectors vec1_D and global_to_D.

.keywords: VecScatters
.seealso: SubdomainSetUp()
@*/
PetscErrorCode SubdomainCreateVecScatter(Subdomain sd, Vec _vec1_global)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  PetscValidHeaderSpecific(_vec1_global,VEC_CLASSID,2);
  ierr = VecDestroy(&sd->vec1_global);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->vec1_D);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)_vec1_global);CHKERRQ(ierr);
  /* Creating work vectors */
  sd->vec1_global = _vec1_global; /* reference to global vector provided by user */
  ierr = VecCreateSeq(PETSC_COMM_SELF,sd->n-sd->n_B,&sd->vec1_D);CHKERRQ(ierr);
  /* Creating the scatter contexts */
  ierr = VecScatterCreate(sd->vec1_global,sd->is_I_global,sd->vec1_D,(IS)0,&sd->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterCreate(sd->vec1_global,sd->is_B_global,sd->vec1_B,(IS)0,&sd->global_to_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
