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
  PetscErrorCode ierr;
  PetscMPIInt    flag, size;
  PetscBool      flg;
  char           hostname[256];

  PetscFunctionBegin;
  if (EinsInitializeCalled) PetscFunctionReturn(0);

  PetscInitialize(argc,args,file,help);
    
  EinsInitializeCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscFinalize"
/*@C
   PetscFinalize - Checks for options to be called at the conclusion
   of the program. MPI_Finalize() is called only if the user had not
   called MPI_Init() before calling PetscInitialize().

   Collective on PETSC_COMM_WORLD

   Options Database Keys:
+  -options_table - Calls PetscOptionsView()
.  -options_left - Prints unused options that remain in the database
.  -objects_dump [all] - Prints list of objects allocated by the user that have not been freed, the option all cause all outstanding objects to be listed
.  -mpidump - Calls PetscMPIDump()
.  -malloc_dump - Calls PetscMallocDump()
.  -malloc_info - Prints total memory usage
-  -malloc_log - Prints summary of memory usage

   Level: beginner

   Note:
   See PetscInitialize() for more general runtime options.

.seealso: PetscInitialize(), PetscOptionsView(), PetscMallocDump(), PetscMPIDump(), PetscEnd()
@*/
PetscErrorCode  EinsFinalize(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       nopt;
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE;
  PetscBool      flg;
#if defined(PETSC_USE_LOG)
  char           mname[PETSC_MAX_PATH_LEN];
#endif

  PetscFunctionBegin;
  if (!PetscInitializeCalled) {
    printf("PetscInitialize() must be called before PetscFinalize()\n");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }

  PetscInitializeCalled = PETSC_FALSE;
  PetscFinalizeCalled   = PETSC_TRUE;
  PetscFunctionReturn(ierr);
}
