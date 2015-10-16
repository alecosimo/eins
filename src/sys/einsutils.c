#include <petsc/private/matimpl.h>


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
