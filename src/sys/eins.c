#include <eins.h>
#include <einsmat.h>
#include <private/einsfetiimpl.h>
#include <private/einstsimpl.h>
#include <private/einskspimpl.h>
#include <private/einspcimpl.h>
#include <private/einsvecimpl.h>
#include <private/einssnesimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/tsimpl.h>
#include <petsc/private/snesimpl.h>

PETSC_EXTERN PetscBool EinsInitializeCalled;
PETSC_EXTERN PetscBool EinsFinalizeCalled;
PETSC_EXTERN PetscBool EinsRegisterAllCalled;

PetscBool EinsInitializeCalled   = PETSC_FALSE;
PetscBool EinsFinalizeCalled     = PETSC_FALSE;
PetscBool EinsRegisterAllCalled  = PETSC_FALSE;

static PetscErrorCode PrintLogDestruction_Private(PetscViewer);

#undef  __FUNCT__
#define __FUNCT__ "EinsRegisterAll"
/*
  Register all methods to KSP, TS, PC and SNES packages  
 */
static PetscErrorCode EinsRegisterAll(void);
static PetscErrorCode EinsRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  EinsRegisterAllCalled = PETSC_TRUE;
  /*Register FETI*/
  ierr = FETIRegisterAll();CHKERRQ(ierr);
  /*Register Vec*/
  ierr = VecRegisterAll();CHKERRQ(ierr);
  ierr = VecRegister(VECMPIUNASM,VecCreate_UNASM);CHKERRQ(ierr);
  /*Register Mat*/
  ierr = MatRegisterAll();CHKERRQ(ierr);
  ierr = MatRegister(MATSHELLUNASM,MatCreate_ShellUnAsm);CHKERRQ(ierr);
  /*Register KSP*/
  ierr = KSPRegisterAll();CHKERRQ(ierr);
  ierr = KSPRegister(KSPPJCG,KSPCreate_PJCG);CHKERRQ(ierr);
  ierr = KSPRegister(KSPPJGMRES,KSPCreate_PJGMRES);CHKERRQ(ierr);
  ierr = KSPRegister(KSPFETI,KSPCreate_FETI);CHKERRQ(ierr);
  /*Register PC*/
  ierr = PCRegisterAll();CHKERRQ(ierr);
  ierr = PCRegister(PCFETI_DIRICHLET,PCCreate_DIRICHLET);CHKERRQ(ierr);
  ierr = PCRegister(PCFETI_LUMPED,PCCreate_LUMPED);CHKERRQ(ierr);
  /*Register TS*/
  ierr = TSRegisterAll();CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA2,TSCreate_Alpha2);CHKERRQ(ierr);
  /*Register SNES*/
  ierr = SNESRegisterAll();CHKERRQ(ierr);
  ierr = SNESRegister(SNESFETIONLY,SNESCreate_FETIONLY);CHKERRQ(ierr);
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
  ierr = PetscClassIdRegister("Vector Exchange",&VEC_EXCHANGE_CLASSID);CHKERRQ(ierr);
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
#define __FUNCT__ "PrintLogDestruction_Private"
/*@
   PrintLogDestruction_Private - Prints information about object
   creation. It is called by EinsFinalize()

   Level: developer

.seealso: EinsFinalize()
@*/
static PetscErrorCode  PrintLogDestruction_Private(PetscViewer viewer)
{
  PetscErrorCode    ierr;
  int                stage, oclass,numStages;
  MPI_Comm           comm;
  FILE               *fd;
  PetscClassPerfInfo *classInfo;
  PetscStageLog      stageLog;
  PetscStageInfo     *stageInfo = NULL;
  PetscBool          *localStageUsed;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&stageLog->numStages, &numStages, 1, MPI_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(numStages, &localStageUsed);CHKERRQ(ierr);
  if (numStages > 0) {
    stageInfo = stageLog->stageInfo;
    for (stage = 0; stage < numStages; stage++) {
      if (stage < stageLog->numStages) {
	localStageUsed[stage]    = stageInfo[stage].used;
      } else {
	localStageUsed[stage]    = PETSC_FALSE;
      }
    }
      
  }

  /* Memory usage and object creation */
  ierr = PetscFPrintf(comm, fd, "------------------------------------------------------------------------------------------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Memory usage is given in bytes:\n\n");CHKERRQ(ierr);

  /* Right now, only stages on the first processor are reported here, meaning only objects associated with
     the global communicator, or MPI_COMM_SELF for proc 1. We really should report global stats and then
     stats for stages local to processor sets.
  */
  /* We should figure out the longest object name here (now 20 characters) */
  ierr = PetscFPrintf(comm, fd, "Object Type          Creations   Destructions\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(comm, fd, "Reports information only for process 0.\n");CHKERRQ(ierr);
  for (stage = 0; stage < numStages; stage++) {
    if (localStageUsed[stage]) {
      classInfo = stageLog->stageInfo[stage].classLog->classInfo;
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: %s\n\n", stage, stageInfo[stage].name);CHKERRQ(ierr);
      for (oclass = 0; oclass < stageLog->stageInfo[stage].classLog->numClasses; oclass++) {
	if ((classInfo[oclass].creations > 0) || (classInfo[oclass].destructions > 0)) {
	  ierr = PetscFPrintf(comm, fd, "%20s %5d          %5d\n", stageLog->classLog->classInfo[oclass].name,
			      classInfo[oclass].creations, classInfo[oclass].destructions);CHKERRQ(ierr);
	}
      }
    } else {
      ierr = PetscFPrintf(comm, fd, "\n--- Event Stage %d: Unknown\n\n", stage);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree(localStageUsed);CHKERRQ(ierr);
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
  PetscBool      petscFinalized,flg;
  PetscErrorCode ierr;
#if defined(PETSC_USE_LOG)
  char           mname[PETSC_MAX_PATH_LEN];
#endif

  PetscFunctionBegin;
  if (!EinsInitializeCalled) {
    printf("EinsInitialize() must be called before EinsFinalize()\n");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  ierr = PetscOptionsGetString(NULL,NULL,"-log_destruction",mname,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    if (mname[0]) {
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,mname,&viewer);CHKERRQ(ierr);
      ierr = PrintLogDestruction_Private(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      viewer = PETSC_VIEWER_STDOUT_WORLD;
      ierr = PrintLogDestruction_Private(viewer);CHKERRQ(ierr);
    }
  }
  
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
