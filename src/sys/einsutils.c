#include <einssys.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/vecimpl.h>
#include <petscmat.h>
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "MatSeqViewSynchronized"
/*@
   MatSeqViewSynchronized - Prints a sequential Mat to the standard
   output in a synchronized form, that is first for process 0, then
   for process 1,... Process 0 waits the last process to
   finish. ACTUALLLY, THIS MUST BE RE-IMPLEMENTED BECAUSE IT IS NOT
   PRINTING IN A SYNCHRONIZED MANNER!!!!!

   Input Parameter:
.  comm   - The MPI communicator 
.  mat    - The sequential matrix to print

   Level: beginner

@*/
PetscErrorCode MatSeqViewSynchronized(MPI_Comm comm,Mat mat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,buff=1;
  MPI_Status     status;
  
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  ierr = MPI_Comm_size(comm,&size);
  ierr = MPI_Comm_rank(comm,&rank);

  if(rank) { ierr = MPI_Recv(&buff,1,MPI_INT,rank-1,0,comm,&status);CHKERRQ(ierr); }
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nMatView process number %d\n",rank);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MPI_Send(&buff,1,MPI_INT,(rank+1)%size,0,comm);CHKERRQ(ierr);
  if(!rank) { ierr = MPI_Recv(&buff,1,MPI_INT,size-1,0,comm,&status);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSeqViewSynchronized"
/*@
   VecSeqViewSynchronized - Prints a sequential Vec to the standard output in a synchronized form, that is first for process 0, then for process 1,... Process 0 waits the last process to finish.

   Input Parameter:
.  comm   - The MPI communicator
.  vec    - The sequential vector to print

   Level: beginner

@*/
PetscErrorCode VecSeqViewSynchronized(MPI_Comm comm,Vec vecprint)
{
  PetscErrorCode    ierr;
  PetscMPIInt       j,n,size,rank;
  PetscInt          work = vecprint->map->n,len;
  MPI_Status        status;
  Vec               vec;
  PetscScalar       *values;
  const PetscScalar *xarray;

  
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(vecprint,VEC_CLASSID,2);
  ierr = MPI_Comm_size(comm,&size);
  ierr = MPI_Comm_rank(comm,&rank);

  ierr = MPI_Reduce(&work,&len,1,MPIU_INT,MPI_MAX,0,comm);CHKERRQ(ierr);
  if(!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Processor # %d out of %d \n",rank,size);CHKERRQ(ierr);
    ierr = VecView(vecprint,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscMalloc1(len,&values);CHKERRQ(ierr);
    for (j=1; j<size; j++) {
      ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,0,comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,values,&vec);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "Processor # %d out of %d \n",j,size);CHKERRQ(ierr);
      ierr = VecView(vec,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      ierr = VecDestroy(&vec);CHKERRQ(ierr);
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(vecprint,&xarray);CHKERRQ(ierr);
    ierr = MPI_Send((void*)xarray,vecprint->map->n,MPIU_SCALAR,0,0,comm);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vecprint,&xarray);CHKERRQ(ierr);
  }    
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


#if defined(PETSC_HAVE_HDF5)
#undef __FUNCT__
#define __FUNCT__ "HDF5ArrayView"
/*@C
    HDF5ArrayView - Writes array to HDF5 file. It is collective in
    PetscViewer, but each processor will write independently its own
    data in array without any communication.

    Collective on PetscViewer

    Input Parameters:
+   N        - number of elements in array
.   array    - array to save in file
.   datatype - HDF5 data type of the elements in array
.   local_sizes  - vector of length N, whose entry i is equal to the number of elements of the array of domain "i". It can be NULL.
-   viewer       - Petsc HDF5 viewer

  Level: intermediate

.seealso: VecView_UNASM_HDF5()
@*/
PETSC_EXTERN PetscErrorCode HDF5ArrayView(PetscInt N,const void* array,hid_t datatype,PetscInt *local_sizes,PetscViewer viewer)
{
  hid_t             filespace; /* file dataspace identifier */
  hid_t             chunkspace; /* chunk dataset property identifier */
  hid_t             plist_id;  /* property list identifier */
  hid_t             dset_id;   /* dataset identifier */
  hid_t             memspace;  /* memory dataspace identifier */
  hid_t             file_id;
  hid_t             group;
  hsize_t           dim;
  hsize_t           maxDims[4], dims[4], chunkDims[4], count[4],offset[4];
  PetscInt          timestep, i;
  PetscErrorCode    ierr;
  PetscBool         alloc=PETSC_FALSE;
  char              vecname_local[10];
  PetscMPIInt       size,rank;
  MPI_Comm          comm;
  
  PetscFunctionBegin;
  ierr = PetscViewerHDF5OpenGroup(viewer, &file_id, &group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetTimestep(viewer, &timestep);CHKERRQ(ierr);

  /* Create the dataset with default properties and close filespace */
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!local_sizes) {
    alloc = PETSC_TRUE;
    ierr = PetscMalloc1(size,&local_sizes);CHKERRQ(ierr);
    ierr = MPI_Allgather(&N,1,MPIU_INT,local_sizes,1,MPIU_INT,comm);CHKERRQ(ierr);
  }
    /* Create the dataspace for the dataset.
     *
     * dims - holds the current dimensions of the dataset
     *
     * maxDims - holds the maximum dimensions of the dataset (unlimited
     * for the number of time steps with the current dimensions for the
     * other dimensions; so only additional time steps can be added).
     *
     * chunkDims - holds the size of a single time step (required to
     * permit extending dataset).
     */

  for (i=0;i<size;i++) {
    
    sprintf(vecname_local,NAMEDOMAIN,i);

    dim = 0;
    if (timestep >= 0) {
      dims[dim]      = timestep+1;
      maxDims[dim]   = H5S_UNLIMITED;
      chunkDims[dim] = 1;
      ++dim;
    }
    ierr = PetscHDF5IntCast(local_sizes[i],dims + dim);CHKERRQ(ierr);

    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    ++dim;

    if (!H5Lexists(group, vecname_local, H5P_DEFAULT)) {
      PetscStackCallHDF5Return(filespace,H5Screate_simple,((int)dim, dims, maxDims));

      /* Create chunk */
      PetscStackCallHDF5Return(chunkspace,H5Pcreate,(H5P_DATASET_CREATE));
      PetscStackCallHDF5(H5Pset_chunk,(chunkspace, (int)dim, chunkDims));

#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
      PetscStackCallHDF5Return(dset_id,H5Dcreate2,(group, vecname_local, datatype, filespace, H5P_DEFAULT, chunkspace, H5P_DEFAULT));
#else
      PetscStackCallHDF5Return(dset_id,H5Dcreate,(group, vecname_local, datatype, filespace, H5P_DEFAULT));
#endif
      PetscStackCallHDF5(H5Pclose,(chunkspace));
      PetscStackCallHDF5(H5Sclose,(filespace));
    } else {
      PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, vecname_local, H5P_DEFAULT));
      PetscStackCallHDF5(H5Dset_extent,(dset_id, dims));
    }
    PetscStackCallHDF5(H5Dclose,(dset_id));
  }

  /* write data to HDF5 file */
  sprintf(vecname_local,NAMEDOMAIN,rank);
  PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, vecname_local, H5P_DEFAULT));

  /* Each process defines a dataset and writes it to the hyperslab in the file */
  dim = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  ierr = PetscHDF5IntCast(N,count + dim);CHKERRQ(ierr);
  ++dim;
  if (N > 0) {
    PetscStackCallHDF5Return(memspace,H5Screate_simple,((int)dim, count, NULL));
  } else {
    /* Can't create dataspace with zero for any dimension, so create null dataspace. */
    PetscStackCallHDF5Return(memspace,H5Screate,(H5S_NULL));
  }

  /* Select hyperslab in the file */
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  ierr = PetscHDF5IntCast(0,offset + dim);CHKERRQ(ierr);
  ++dim;
  if (N > 0) {
    PetscStackCallHDF5Return(filespace,H5Dget_space,(dset_id));
    PetscStackCallHDF5(H5Sselect_hyperslab,(filespace, H5S_SELECT_SET, offset, NULL, count, NULL));
  } else {
    /* Create null filespace to match null memspace. */
    PetscStackCallHDF5Return(filespace,H5Screate,(H5S_NULL));
  }

  /* Create property list for collective dataset write */
  PetscStackCallHDF5Return(plist_id,H5Pcreate,(H5P_DATASET_XFER));
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  PetscStackCallHDF5(H5Pset_dxpl_mpio,(plist_id, H5FD_MPIO_INDEPENDENT));
#endif

  PetscStackCallHDF5(H5Dwrite,(dset_id, datatype, memspace, filespace, plist_id, array));
  PetscStackCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL));

  /* Close/release resources */
  if (group != file_id) {
    PetscStackCallHDF5(H5Gclose,(group));
  }
  PetscStackCallHDF5(H5Pclose,(plist_id));
  PetscStackCallHDF5(H5Sclose,(filespace)); 
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));
  if (alloc) { ierr = PetscFree(local_sizes);CHKERRQ(ierr);}
  ierr   = PetscInfo(viewer,"Wrote array in HDF5 file\n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif
