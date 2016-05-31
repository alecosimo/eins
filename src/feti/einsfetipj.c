#include <private/einsfetiimpl.h>
#include <private/einsmatimpl.h>
#include <einssys.h>

PetscClassId      FETIPJ_CLASSID;
PetscLogEvent     FETIPJ_SetUp;
PetscBool         FETIPJRegisterAllCalled   = PETSC_FALSE;
PetscFunctionList FETIPJList                = 0;

PETSC_EXTERN PetscErrorCode FETIPJCreate_PJ2LEVEL(FETIPJ);
PETSC_EXTERN PetscErrorCode FETIPJCreate_PJNONE(FETIPJ);


#undef __FUNCT__
#define __FUNCT__ "FETIPJRegister"
/*@C
  FETIPJRegister -  Adds a FETI projection.

   Not collective

   Input Parameters:
+  name_feti - name of a new user-defined FETI projection
-  routine_create - routine to create FETI projection context

   Level: advanced

.keywords: FETIPJ

.seealso: FETIPJRegisterAll()
@*/
PetscErrorCode  FETIPJRegister(const char sname[],PetscErrorCode (*function)(FETIPJ))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&FETIPJList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJRegisterAll"
/*@C
   FETIPJRegisterAll - Registers all of the FETI projection in the FETI package.

   Not Collective

   Level: advanced

.keywords: FETIPJ

.seealso: FETIPJRegister()
@*/
PetscErrorCode  FETIPJRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (FETIPJRegisterAllCalled) PetscFunctionReturn(0);
  FETIPJRegisterAllCalled = PETSC_TRUE;

  ierr = FETIPJRegister(PJ_NONE,FETIPJCreate_PJNONE);CHKERRQ(ierr);
  ierr = FETIPJRegister(PJ_SECOND_LEVEL,FETIPJCreate_PJ2LEVEL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJSetType"
/*@C
   FETIPJSetType - Builds FETIPJ for the particular FETIPJ type

   Collective on FETIPJ

   Input Parameter:
+  ftpj - the FETIPJ context
-  type - a FETI projection

   Options Database Key:
.  -fetipj_type <type> - Sets FETIPJ type

  Level: intermediate

.seealso: FETIPJType, FETIPJRegister(), FETIPJCreate()

@*/
PetscErrorCode  FETIPJSetType(FETIPJ ftpj,const FETIPJType type)
{
  PetscErrorCode ierr,(*func)(FETIPJ);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)ftpj,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(FETIPJList,type,&func);CHKERRQ(ierr);
  if (!func) SETERRQ1(PetscObjectComm((PetscObject)ftpj),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested FETIPJ type %s",type);
  /* Destroy the previous private FETIPJ context */
  if (ftpj->ops->destroy) {
    ierr               =  (*ftpj->ops->destroy)(ftpj);CHKERRQ(ierr);
    ftpj->ops->destroy = NULL;
    ftpj->data         = 0;
  }
  ierr = PetscFunctionListDestroy(&((PetscObject)ftpj)->qlist);CHKERRQ(ierr);
  /* Reinitialize function pointers in FETIPJOps structure */
  ierr = PetscMemzero(ftpj->ops,sizeof(struct _FETIOps));CHKERRQ(ierr);
  /* Call the FETIPJCreate_XXX routine for this particular FETI formulation */
  ierr       = PetscObjectChangeTypeName((PetscObject)ftpj,type);CHKERRQ(ierr);
  ierr       = (*func)(ftpj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJSetFromOptions"
/*@
   FETIPJSetFromOptions - Sets FETIPJ options from the options database.
   This routine must be called before FETIPJSetUp().

   Collective on FETIPJ

   Input Parameter:
.  ftpj - the FETIPJ context

   Options Database:
.  -fetipj_type: speciefies the FETIPJ projection

   Level: begginer

.keywords: FETIPJ

@*/
PetscErrorCode  FETIPJSetFromOptions(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  char           type[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);

  ierr = FETIPJRegisterAll();CHKERRQ(ierr);
  if (!ftpj->feti) SETERRQ(PetscObjectComm((PetscObject)ftpj),PETSC_ERR_ARG_WRONGSTATE,"Error FETI context not defined");
  ierr = PetscObjectOptionsBegin((PetscObject)ftpj);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-fetipj_type","FETIPJ","FETIPJSetType",FETIPJList,ftpj->feti->ftpj_type,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = FETIPJSetType(ftpj,type);CHKERRQ(ierr);
    ftpj->feti->ftpj_type = ((PetscObject)ftpj)->type_name;
  } else if (!((PetscObject)ftpj)->type_name) {
    ierr = FETIPJSetType(ftpj,ftpj->feti->ftpj_type);CHKERRQ(ierr);
  }
  if (ftpj->ops->setfromoptions) {
    ierr = (*ftpj->ops->setfromoptions)(PetscOptionsObject,ftpj);CHKERRQ(ierr);
  }

  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)ftpj);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJDestroy"
/*@
   FETIPJDestroy - Destroys FETIPJ context that was created with FETIPJCreate().

   Collective on FETIPJ

   Input Parameter:
.  feti - the FETIPJ context

   Level: developer

.keywords: FETIPJ

.seealso: FETIPJCreate(), FETIPJSetUp()
@*/
PetscErrorCode FETIPJDestroy(FETIPJ *_ftpj)
{
  PetscErrorCode ierr;
  FETIPJ         ftpj;

  PetscFunctionBegin;
  PetscValidPointer(_ftpj,1);
  ftpj = *_ftpj; *_ftpj = NULL;
  if (!ftpj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  if (--((PetscObject)ftpj)->refct > 0) PetscFunctionReturn(0);

  if (ftpj->ops->destroy) {ierr = (*ftpj->ops->destroy)(ftpj);CHKERRQ(ierr);}  

  ierr = PetscHeaderDestroy(&ftpj);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJSetUp"
/*@
   FETIPJSetUp - Prepares the structures needed by FETIPJ.

   Collective on FETIPJ

   Input Parameter:
.  feti - the FETIPJ context

   Level: developer

.keywords: FETIPJ

.seealso: FETIPJCreate(), FETIPJDestroy()
@*/
PetscErrorCode  FETIPJSetUp(FETIPJ ftpj)
{
  PetscErrorCode   ierr;
  const char*      def = "pjnone";

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  if (ftpj->setupcalled) PetscFunctionReturn(0);
  if (!ftpj->feti) SETERRQ(PetscObjectComm((PetscObject)ftpj),PETSC_ERR_ARG_WRONGSTATE,"Error FETI context not defined");
  if (!ftpj->setupcalled) { ierr = PetscInfo(ftpj,"Setting up FETIPJ for first time\n");CHKERRQ(ierr);} 
  if (!((PetscObject)ftpj)->type_name) {
    ierr = FETIPJSetType(ftpj,def);CHKERRQ(ierr);
    ftpj->feti->ftpj_type = def;
  }

  if (ftpj->ops->setup) {
    ierr = (*ftpj->ops->setup)(ftpj);CHKERRQ(ierr);
  }
  if (!ftpj->setupcalled) ftpj->setupcalled++;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETIPJGatherNeighborsCoarseBasis"
/*@
   FETIPJGatherNeighborsCoarseBasis - Gathers the coarse basis matrix
   at the interface dofs of the subdomain's neigbors.

   Input: 
.  ftpj - the FETIPJ context

   Level: basic

.keywords: FETIPJ

.seealso: FETIPJAssembleCoarseProblem(),FETIPJFactorizeCoarseProblem()
@*/
PetscErrorCode FETIPJGatherNeighborsCoarseBasis(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  if(ftpj->state==FETIPJ_STATE_NEIGH_GATHERED || ftpj->state==FETIPJ_STATE_ASSEMBLED) PetscFunctionReturn(0);
  if (ftpj->ops->gatherneighbors) { ierr = (*ftpj->ops->gatherneighbors)(ftpj);CHKERRQ(ierr); }
  ftpj->state = FETIPJ_STATE_NEIGH_GATHERED;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETIPJAssembleCoarseProblem"
/*@
   FETIPJAssembleCoarseProblem - Assembles the coarse problem.

   Input: 
.  ftpj - the FETIPJ context

   Level: basic

.keywords: FETIPJ

.seealso: FETIPJGatherNeighborsCoarseBasis(),FETIPJFactorizeCoarseProblem()
@*/
PetscErrorCode FETIPJAssembleCoarseProblem(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  if(ftpj->state!=FETIPJ_STATE_NEIGH_GATHERED) PetscFunctionReturn(0);
  if (ftpj->ops->assemble) { ierr = (*ftpj->ops->assemble)(ftpj);CHKERRQ(ierr); }
  ftpj->state = FETIPJ_STATE_ASSEMBLED;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETIPJComputeInitialCondition"
/*@
   FETIPJComputeInitialCondition - Computes initial condition
   for the FETI interface problem.

   Input: 
.  ftpj - the FETIPJ context

   Level: basic

.keywords: FETIPJ

.seealso: FETIPJGatherNeighborsCoarseBasis(),FETIPJFactorizeCoarseProblem() FETIPJAssembleCoarseProblem()
@*/
PetscErrorCode FETIPJComputeInitialCondition(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  if(ftpj->state<FETIPJ_STATE_FACTORIZED) PetscFunctionReturn(0);
  if (ftpj->ops->initialcondition) {
    ierr = (*ftpj->ops->initialcondition)(ftpj);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)ftpj),PETSC_ERR_ARG_WRONGSTATE,"Error: FETIPJComputeInitialCondition of specific FETIPJ method not found.");
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETIPJFactorizeCoarseProblem"
/*@
   FETIPJFactorizeCoarseProblem - Assembles the coarse problem.

   Input: 
.  ftpj - the FETIPJ context

   Level: basic

.keywords: FETIPJ

.seealso: FETIPJAssembleCoarseProblem(),FETIPJGatherNeighborsCoarseBasis()
@*/
PetscErrorCode FETIPJFactorizeCoarseProblem(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  if(ftpj->state!=FETIPJ_STATE_ASSEMBLED) PetscFunctionReturn(0);
  if (ftpj->ops->factorize) { ierr = (*ftpj->ops->factorize)(ftpj);CHKERRQ(ierr); }
  ftpj->state = FETIPJ_STATE_FACTORIZED;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETIPJSetFETI"
/*@
   FETIPJSetFETI - Sets the FETI context which will make use of the current feti projection.

   Input: 
.  ftpj - the FETIPJ context
.  feti - the FETI context

   Level: basic

.keywords: FETIPJ, FETI

.seealso: FETIPJSetUp, FETIPJSetType
@*/
PetscErrorCode FETIPJSetFETI(FETIPJ ftpj,FETI feti)
{  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETIPJ_CLASSID,1);
  PetscValidHeaderSpecific(feti,FETI_CLASSID,2);
  ftpj->feti = feti;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJGetType"
/*@C
   FETIPJGetType - Gets the FETIPJ type and name (as a string) the FETIPJ context.

   Not Collective

   Input Parameter:
.  ftpj - the FETIPJ context

   Output Parameter:
.  type - name of FETIPJ

   Level: intermediate

.keywords: FETI

.seealso: FETIPJSetType()

@*/
PetscErrorCode FETIPJGetType(FETIPJ ftpj,FETIPJType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftpj,FETI_CLASSID,1);
  *type = ((PetscObject)ftpj)->type_name;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJCreate"
/*@
   FETIPJCreate - Creates a FETIPJ context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  ftpj - location to put the FETIPJ context
.  feti - the FETI context.

   Options Database:
.  -fetipj_type <type> - Sets the FETIPJ type

   Level: developer

.keywords: FETIPJ

.seealso: FETIPJSetUp(), FETIPJDestroy()
@*/
PetscErrorCode  FETIPJCreate(MPI_Comm comm,FETI feti,FETIPJ *newftpj)
{
  FETIPJ         ftpj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newftpj,1);
  PetscValidHeaderSpecific(feti,FETI_CLASSID,2);
  *newftpj = 0;
  ierr = FETIInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(ftpj,FETIPJ_CLASSID,"FETIPJ","FETIPJ","FETIPJ",comm,FETIPJDestroy,NULL);CHKERRQ(ierr);
  ftpj->state       = FETIPJ_STATE_INITIAL;
  ftpj->setupcalled = 0;
  ftpj->feti        = feti;
  
  *newftpj = ftpj;
  PetscFunctionReturn(0);

}
