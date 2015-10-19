#include <petsc/private/matimpl.h>
#include <petscsf.h>


#undef __FUNCT__
#define __FUNCT__ "VecSeqViewSynchronized"
/*@
   VecSeqViewSynchronized - Prints a sequential Vec to the standard output in a synchronized form, that is first for process 0, then for process 1,...

   Input Parameter:
.  vec    - The sequential vector to print

   Level: beginner

@*/
PetscErrorCode VecSeqViewSynchronized(Vec vec)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,buff=1;
  MPI_Comm       comm = MPI_COMM_WORLD;
  MPI_Status     status;
  
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = MPI_Comm_size(comm,&size);
  ierr = MPI_Comm_rank(comm,&rank);

  if(rank) { ierr = MPI_Recv(&buff,1,MPI_INT,rank-1,0,comm,&status);CHKERRQ(ierr); }
  PetscPrintf(PETSC_COMM_SELF,"\nVecView process number %d\n",rank);
  VecView(vec,PETSC_VIEWER_STDOUT_SELF);
  if(rank+1<size) { ierr = MPI_Send(&buff,1,MPI_INT,rank+1,0,comm);CHKERRQ(ierr); }
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ISCreateMPIVec"
/*@
   ISCreateGlobalVec - Creates a distributed vector with a specified global size and global to local numbering 

   Input Parameter:
.  comm        - The MPI communicator
.  global_size - The global size of the distributed vector
.  mapping     - The loval to global mapping

   Output Parameter:
.  _vec         - The created distributed vector

   Level: intermediate

@*/
PetscErrorCode ISCreateMPIVec(MPI_Comm comm,PetscInt global_size,ISLocalToGlobalMapping mapping,Vec *_vec)
{
  Vec            vec;
  PetscErrorCode ierr;
  
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,3);
  ierr  = VecCreate(comm,&vec);CHKERRQ(ierr);
  ierr  = VecSetType(vec,VECMPI);CHKERRQ(ierr);
  ierr  = VecSetSizes(vec,PETSC_DECIDE,global_size);CHKERRQ(ierr);
  ierr  = VecSetLocalToGlobalMapping(vec,mapping);CHKERRQ(ierr);
  *_vec = vec;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ISSubsetNumbering"
/*@
   ISSubsetNumbering - Given an index set (IS) possibly with holes, renumbers the indexes removing the holes

   Input Parameter:
.  subset      - IS which is a subset of a numbering. In the general this IS will have holes.
.  subset_mult - IS with the multiplicity of each entry of subset. 

   Output Parameter:
.  N_n       - Number of entries in the new renumbered IS.
.  subset_n  - Renumbered IS without holes taking into account the multiplicity.

   Level: intermediate

@*/
PetscErrorCode ISSubsetNumbering(IS subset, IS subset_mult, PetscInt *N_n, IS *subset_n)
{
  PetscSF        sf;
  PetscLayout    map;
  const PetscInt *idxs;
  PetscInt       *leaf_data,*root_data,*gidxs;
  PetscInt       N,n,i,lbounds[2],gbounds[2],Nl;
  PetscInt       n_n,nlocals,start,first_index;
  PetscMPIInt    commsize;
  PetscBool      first_found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(subset,&n);CHKERRQ(ierr);
  if (subset_mult) {
    PetscCheckSameComm(subset,1,subset_mult,2);
    ierr = ISGetLocalSize(subset,&i);CHKERRQ(ierr);
    if (i != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local subset and multiplicity sizes don't match! %d != %d",n,i);
  }
  /* create workspace layout for computing global indices of subset */
  ierr = ISGetIndices(subset,&idxs);CHKERRQ(ierr);
  lbounds[0] = lbounds[1] = 0;
  for (i=0;i<n;i++) {
    if (idxs[i] < lbounds[0]) lbounds[0] = idxs[i];
    else if (idxs[i] > lbounds[1]) lbounds[1] = idxs[i];
  }
  lbounds[0] = -lbounds[0];
  ierr = MPI_Allreduce(lbounds,gbounds,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
  gbounds[0] = -gbounds[0];
  N = gbounds[1] - gbounds[0] + 1;
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)subset),&map);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(map,N);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map,&Nl);CHKERRQ(ierr);

  /* create sf : leaf_data == multiplicity of indexes, root data == global index in layout */
  ierr = PetscMalloc2(n,&leaf_data,Nl,&root_data);CHKERRQ(ierr);
  if (subset_mult) {
    const PetscInt* idxs_mult;

    ierr = ISGetIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
    ierr = PetscMemcpy(leaf_data,idxs_mult,n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
  } else {
    for (i=0;i<n;i++) leaf_data[i] = 1;
  }
  /* local size of new subset */
  n_n = 0;
  for (i=0;i<n;i++) n_n += leaf_data[i];

  /* global indexes in layout */
  ierr = PetscMalloc1(n_n,&gidxs);CHKERRQ(ierr); /* allocating possibly extra space in gidxs which will be used later */
  for (i=0;i<n;i++) gidxs[i] = idxs[i] - gbounds[0];
  ierr = ISRestoreIndices(subset,&idxs);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)subset),&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sf,map,n,NULL,PETSC_COPY_VALUES,gidxs);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);

  /* reduce from leaves to roots */
  ierr = PetscMemzero(root_data,Nl*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,leaf_data,root_data,MPI_MAX);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,leaf_data,root_data,MPI_MAX);CHKERRQ(ierr);

  /* count indexes in local part of layout */
  nlocals = 0;
  first_index = -1;
  first_found = PETSC_FALSE;
  for (i=0;i<Nl;i++) {
    if (!first_found && root_data[i]) {
      first_found = PETSC_TRUE;
      first_index = i;
    }
    nlocals += root_data[i];
  }

  /* cumulative of number of indexes and size of subset without holes */
#if defined(PETSC_HAVE_MPI_EXSCAN)
  start = 0;
  ierr = MPI_Exscan(&nlocals,&start,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
#else
  ierr = MPI_Scan(&nlocals,&start,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
  start = start-nlocals;
#endif

  if (N_n) { /* compute total size of new subset if requested */
    *N_n = start + nlocals;
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)subset),&commsize);CHKERRQ(ierr);
    ierr = MPI_Bcast(N_n,1,MPIU_INT,commsize-1,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
  }

  /* adapt root data with cumulative */
  if (first_found) {
    PetscInt old_index;

    root_data[first_index] += start;
    old_index = first_index;
    for (i=first_index+1;i<Nl;i++) {
      if (root_data[i]) {
        root_data[i] += root_data[old_index];
        old_index = i;
      }
    }
  }

  /* from roots to leaves */
  ierr = PetscSFBcastBegin(sf,MPIU_INT,root_data,leaf_data);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,root_data,leaf_data);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* create new IS with global indexes without holes */
  if (subset_mult) {
    const PetscInt* idxs_mult;
    PetscInt        cum;

    cum = 0;
    ierr = ISGetIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      PetscInt j;
      for (j=0;j<idxs_mult[i];j++) gidxs[cum++] = leaf_data[i] - idxs_mult[i] + j;
    }
    ierr = ISRestoreIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
  } else {
    for (i=0;i<n;i++) {
      gidxs[i] = leaf_data[i]-1;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)subset),n_n,gidxs,PETSC_OWN_POINTER,subset_n);CHKERRQ(ierr);
  ierr = PetscFree2(leaf_data,root_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
