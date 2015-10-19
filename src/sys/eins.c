#include <eins.h>
#include <private/einsfetiimpl.h>
#include <private/einskspimpl.h>


PETSC_EXTERN PetscBool EinsInitializeCalled;
PETSC_EXTERN PetscBool EinsFinalizeCalled;
PETSC_EXTERN PetscBool EinsRegisterAllCalled;

PetscBool EinsInitializeCalled   = PETSC_FALSE;
PetscBool EinsFinalizeCalled     = PETSC_FALSE;
PetscBool EinsRegisterAllCalled  = PETSC_FALSE;


#undef  __FUNCT__
#define __FUNCT__ "EinsRegisterAll"
/*
  Register all methods to KSP, TS, PC and SNES packages  
 */
PetscErrorCode EinsRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  EinsRegisterAllCalled = PETSC_TRUE;
  /*Register FETI*/
  ierr = FETIRegisterAll();CHKERRQ(ierr);
  /*Register KSP*/
  ierr = KSPRegisterAll();CHKERRQ(ierr);
  ierr = KSPRegister(KSPPJGMRES,KSPCreate_PJGMRES);CHKERRQ(ierr);
  /*Register PC*/
  /*Register TS*/
  /*Register SNES*/
  PetscFunctionReturn(0);
}


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
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (EinsInitializeCalled) PetscFunctionReturn(0);
  ierr = PetscInitialized(&petscCalled);CHKERRQ(ierr);
  if(!petscCalled)  {ierr = PetscInitialize(argc,args,file,help);CHKERRQ(ierr);}
  /* Register Classes */
  /*-- */
  /* Register Constructors */
  ierr = EinsRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  /*-- */
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
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (!EinsInitializeCalled) {
    printf("EinsInitialize() must be called before EinsFinalize()\n");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  if (KSPList) {ierr = PetscFunctionListDestroy(&KSPList);CHKERRQ(ierr);}
  ierr = PetscFinalized(&petscFinalized);CHKERRQ(ierr);
  if(!petscFinalized)  {ierr = PetscFinalize();CHKERRQ(ierr);}   
  EinsInitializeCalled  = PETSC_FALSE;
  EinsFinalizeCalled    = PETSC_TRUE;
  EinsRegisterAllCalled = PETSC_FALSE;
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
