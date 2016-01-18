#include <einssys.h>
#include <petsc/private/petscimpl.h>


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
.   local_sizes  - pointer to a pointer which defines a vector of length N, whose entry i 
    is equal to the number of elements of the array of domain "i". If the double pointer 
    local_sizes is NULL, space is allocated and freed by the routine. If the pointer to 
    the pointer is not NULL, but the referenced pointer is NULL space is allocated but not
    freed by the routine and the user must do it.
-   viewer       - Petsc HDF5 viewer

  Level: intermediate

.seealso: VecView_UNASM_HDF5()
@*/
PETSC_EXTERN PetscErrorCode HDF5ArrayView(PetscInt N,const void* array,hid_t datatype,PetscInt **local_sizes,PetscViewer viewer)
{
  hid_t             filespace; /* file dataspace identifier */
  hid_t             chunkspace; /* chunk dataset property identifier */
  hid_t             plist_id;  /* property list identifier */
  hid_t             dset_id;   /* dataset identifier */
  hid_t             memspace;  /* memory dataspace identifier */
  hid_t             file_id;
  hid_t             group;
  hsize_t           dim;
  hsize_t           maxDims[4],dims[4],chunkDims[4],count[4],offset[4];
  PetscInt          timestep,i,*ref;
  PetscErrorCode    ierr;
  PetscBool         dealloc=PETSC_FALSE;
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
    dealloc = PETSC_TRUE;
    ierr    = PetscMalloc1(size,&ref);CHKERRQ(ierr);
    local_sizes = &ref;
    ierr = MPI_Allgather(&N,1,MPIU_INT,*local_sizes,1,MPIU_INT,comm);CHKERRQ(ierr);
  } else if (!(*local_sizes)) {
    ierr = PetscMalloc1(size,local_sizes);CHKERRQ(ierr);
    ierr = MPI_Allgather(&N,1,MPIU_INT,*local_sizes,1,MPIU_INT,comm);CHKERRQ(ierr);    
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
    ierr = PetscHDF5IntCast((*local_sizes)[i],dims + dim);CHKERRQ(ierr);

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
  /*  PetscStackCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL)); */

  /* Close/release resources */
  if (group != file_id) {
    PetscStackCallHDF5(H5Gclose,(group));
  }
  PetscStackCallHDF5(H5Pclose,(plist_id));
  PetscStackCallHDF5(H5Sclose,(filespace)); 
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));
  if (dealloc) { ierr = PetscFree(*local_sizes);CHKERRQ(ierr);}
  ierr   = PetscInfo(viewer,"Wrote array in HDF5 file\n");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif


#if defined(PETSC_HAVE_HDF5)
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5WriteGroupAttribute"
/*@
 PetscViewerHDF5WriteGroupAttribute - Write a scalar attribute to a group

  Input Parameters:
+ viewer     - The HDF5 viewer
. name       - The attribute name
. datatype   - The attribute type
- value      - The attribute value

  Level: advanced

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5ReadAttribute(), PetscViewerHDF5HasAttribute()
@*/
PetscErrorCode PetscViewerHDF5WriteGroupAttribute(PetscViewer viewer, const char name[], PetscDataType datatype, const void *value)
{
  hid_t          h5, dataspace, group, attribute, dtype;
  PetscErrorCode ierr;
  htri_t         hhas;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(name, 3);
  PetscValidPointer(value, 4);
  ierr = PetscDataTypeToHDF5DataType(datatype, &dtype);CHKERRQ(ierr);
  if (datatype == PETSC_STRING) {
    size_t len;
    ierr = PetscStrlen((const char *) value, &len);CHKERRQ(ierr);
    PetscStackCallHDF5(H5Tset_size,(dtype, len+1));
  }
  ierr = PetscViewerHDF5OpenGroup(viewer, &h5, &group);CHKERRQ(ierr);
  PetscStackCallHDF5Return(dataspace,H5Screate,(H5S_SCALAR)); 
  PetscStackCall("H5Aexists",hhas = H5Aexists(group, name));
  if(!hhas) {
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
    PetscStackCallHDF5Return(attribute,H5Acreate2,(group, name, dtype, dataspace, H5P_DEFAULT, H5P_DEFAULT));
#else
    PetscStackCallHDF5Return(attribute,H5Acreate,(group, name, dtype, dataspace, H5P_DEFAULT));
#endif
  } else {
    PetscStackCallHDF5Return(attribute,H5Aopen_name,(group, name));
  }
  PetscStackCallHDF5(H5Awrite,(attribute, dtype, value));
  if (datatype == PETSC_STRING) PetscStackCallHDF5(H5Tclose,(dtype));
  if (group != h5) {
    PetscStackCallHDF5(H5Gclose,(group));
  }
  PetscStackCallHDF5(H5Aclose,(attribute));
  PetscStackCallHDF5(H5Sclose,(dataspace));
  PetscFunctionReturn(0);
}
#endif


#if defined(PETSC_HAVE_HDF5)
#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5CreateSoftLink"
/*@
 PetscViewerHDF5CreateSoftLink - Creates an HDF5 soft link

  Input Parameters:
+ viewer       - The HDF5 viewer
. target_path  - The target path relative to the current HDF5 group for output
- link_name    - The link name relative to the current HDF5 group for output

  Level: advanced

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5ReadAttribute(), PetscViewerHDF5HasAttribute()
@*/
PetscErrorCode PetscViewerHDF5CreateSoftLink(PetscViewer viewer, const char target_path[], const char link_name[])
{
  hid_t          h5, group;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidPointer(target_path, 2);
  PetscValidPointer(link_name, 3);
  ierr = PetscViewerHDF5OpenGroup(viewer, &h5, &group);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Lcreate_soft,(target_path, group, link_name, H5P_DEFAULT, H5P_DEFAULT));
  if (group != h5) {
    PetscStackCallHDF5(H5Gclose,(group));
  }
  PetscFunctionReturn(0);
}
#endif

