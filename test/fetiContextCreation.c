static char help[] = "Test the FETI context creation\n\n";

#include <eins.h>
#include <einsfeti.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  MPI_Comm           comm;
  PetscErrorCode     ierr;
  FETI               feti;
  
  ierr = EinsInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = MPI_COMM_SELF;

  ierr = FETICreate(comm,&feti);CHKERRQ(ierr);
  ierr = FETISetType(feti,FETI1);CHKERRQ(ierr);
  ierr = FETIDestroy(&feti);CHKERRQ(ierr);
  
  ierr = EinsFinalize();
  return 0;
}
