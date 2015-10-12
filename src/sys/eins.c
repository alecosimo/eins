#include <petscsys.h>

/* ------------------------ Global variables -------------------------------*/
PetscBool   EinsInitializeCalled = PETSC_FALSE;
PetscBool   EinsFinalizeCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "EinsInitialize"
/*@C EinsInitialize - Initializes Eins and calls to PetscInitialize().
   PetscInitialize() calls MPI_Init(). This function mainly extends
   PetscInitialize() by registering in PETSc new solvers and
   preconditioners. The following description of the Input and Output
   parameters are inhereted from the PETSc implementation. These
   parameters are passed to PetscInitialize().
 
   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - Optional PETSc database file
-  help - Optional PETSc help message to print, use NULL for no message


   Level: beginner


.seealso: EinsFinalize()

@*/
PetscErrorCode  EinsInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscBool petscCalled;
  
  PetscFunctionBegin;
  if (EinsInitializeCalled) PetscFunctionReturn(0);
  ierr = PetscInitialized(&petscCalled);CHKERRQ(ierr);
  if(!petscCalled)  {ierr = PetscInitialize(argc,args,file,help);CHKERRQ(ierr);}
    
  EinsInitializeCalled = PETSC_TRUE;
  EinsFinalizeCalled   = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "EinsFinalize"
/*@C
   EinsFinalize - Finalizes the program. Calls to PetscFinalize().

   Collective on PETSC_COMM_WORLD

   Level: beginner

.seealso: EinsInitialize()
@*/
PetscErrorCode  EinsFinalize(void)
{
  PetscBool petscFinalized;
  
  PetscFunctionBegin;
  if (!EinsInitializeCalled) {
    printf("EinsInitialize() must be called before EinsFinalize()\n");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  ierr = PetscFinalized(petscFinalized);CHKERRQ(ierr);
  if(!petscFinalized)  {ierr = PetscFinalize();CHKERRQ(ierr);}
   
  EinsInitializeCalled = PETSC_FALSE;
  EinsFinalizeCalled   = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EinsInitialized"
/*@
   EinsInitialized - Determine whether EINS is initialized.

   Level: beginner

.seealso: EinsInitialize()
@*/
PetscErrorCode EinsInitialized(PetscBool  *isInitialized)
{
  *isInitialized = EinsInitializeCalled;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "EinsFinalized"
/*@
      EinsFinalized - Determine whether EinsFinalize() has been called yet

   Level: developer

.seealso: EinsInitialize()
@*/
PetscErrorCode  EinsFinalized(PetscBool  *isFinalized)
{
  *isFinalized = EinsFinalizeCalled;
  return 0;
}
