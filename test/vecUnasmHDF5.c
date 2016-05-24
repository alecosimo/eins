static char help[] = "Test the creation of globally unassembled vector and write it to HDF5 file\n\n";

#include <eins.h>
#include <einstest.h>
#include <petscviewerhdf5.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  MPI_Comm               comm;
  PetscInt               rank,idx[5]={0,1,2,3,4};
  PetscScalar            vals[5];
  PetscErrorCode         ierr;
  Vec                    v;
  PetscViewer            viewer;
  
  ierr = EinsInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = MPI_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecCreate(comm,&v);CHKERRQ(ierr);
  ierr = VecSetType(v,VECMPIUNASM);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)v, "CONNECTIVITY");CHKERRQ(ierr);
  switch (rank){
  case 0:
    ierr = VecSetSizes(v,5,PETSC_DECIDE);CHKERRQ(ierr);
    vals[0]=1;vals[1]=2;vals[2]=3;vals[3]=4;vals[4]=5;
    ierr = VecSetValuesLocal(v,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecSetSizes(v,4,PETSC_DECIDE);CHKERRQ(ierr);
    vals[0]=6;vals[1]=7;vals[2]=8;vals[3]=9;
    ierr = VecSetValuesLocal(v,4,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecSetSizes(v,3,PETSC_DECIDE);CHKERRQ(ierr);
    vals[0]=10;vals[1]=11;vals[2]=12;
    ierr = VecSetValuesLocal(v,3,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    break;
  }
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
 
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

#if defined(PETSC_HAVE_HDF5)  
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"results.h5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer,"/");CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, 0);CHKERRQ(ierr);
  ierr = VecView(v,viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, 1);CHKERRQ(ierr);
  ierr = VecScale(v,3);CHKERRQ(ierr);
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* ierr = VecView(v,viewer);CHKERRQ(ierr); */

  ierr = PetscViewerHDF5CreateSoftLink(viewer,"/CONNECTIVITY","/LINK_CONNECTIVITY");CHKERRQ(ierr);
  
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#endif
  
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = EinsFinalize();CHKERRQ(ierr);
  return 0;
}
