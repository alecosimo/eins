#include <private/einsfetiimpl.h>
#include <private/einsmatimpl.h>
#include <einssys.h>

PetscClassId      FETICS_CLASSID;
PetscLogEvent     FETICS_SetUp;
PetscBool         FETICSRegisterAllCalled   = PETSC_FALSE;
PetscFunctionList FETICSList                = 0;

PETSC_EXTERN PetscErrorCode FETICSCreate_RBM(FETICS);
PETSC_EXTERN PetscErrorCode FETICSCreate_GENEO(FETICS);
#if !defined(HAVE_SLEPC)
#undef __FUNCT__
#define __FUNCT__ "FETICSCreate_GENEO"
PetscErrorCode FETICSCreate_GENEO(FETICS ftcs)
{
  PetscFunctionBegin;
  ftcs->data = 0;
  ftcs->ops->setup               = 0;
  ftcs->ops->destroy             = 0;
  ftcs->ops->setfromoptions      = 0;
  ftcs->ops->computecoarsebasis  = 0;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "FETICSCreate_NOCS"
PETSC_EXTERN PetscErrorCode FETICSCreate_NOCS(FETICS);
PetscErrorCode FETICSCreate_NOCS(FETICS ftcs)
{
  PetscFunctionBegin;
  ftcs->data = 0;
  ftcs->ops->setup               = 0;
  ftcs->ops->destroy             = 0;
  ftcs->ops->setfromoptions      = 0;
  ftcs->ops->computecoarsebasis  = 0;

  PetscFunctionReturn(0);  
}

#undef __FUNCT__
#define __FUNCT__ "FETICSRegister"
/*@C
  FETICSRegister -  Adds a FETI coarse space.

   Not collective

   Input Parameters:
+  name_feti - name of a new user-defined FETI coarse space
-  routine_create - routine to create FETI coarse space context

   Level: advanced

.keywords: FETICS

.seealso: FETICSRegisterAll()
@*/
PetscErrorCode  FETICSRegister(const char sname[],PetscErrorCode (*function)(FETICS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&FETICSList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSRegisterAll"
/*@C
   FETICSRegisterAll - Registers all of the FETI coarse spaces in the FETI package.

   Not Collective

   Level: advanced

.keywords: FETICS

.seealso: FETICSRegister()
@*/
PetscErrorCode  FETICSRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (FETICSRegisterAllCalled) PetscFunctionReturn(0);
  FETICSRegisterAllCalled = PETSC_TRUE;

  ierr = FETICSRegister(CS_RIGID_BODY_MODES,FETICSCreate_RBM);CHKERRQ(ierr);
  ierr = FETICSRegister(CS_GENEO_MODES,FETICSCreate_GENEO);CHKERRQ(ierr);
  ierr = FETICSRegister(CS_NONE,FETICSCreate_NOCS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSSetType"
/*@C
   FETICSSetType - Builds FETICS for the particular FETICS type

   Collective on FETICS

   Input Parameter:
+  ftcs - the FETICS context
-  feti - the FETI context.
-  type - a FETI coarse space

   Options Database Key:
.  -feti_type <type> - Sets FETI type

  Level: intermediate

.seealso: FETICSType, FETICSRegister(), FETICSCreate()

@*/
PetscErrorCode  FETICSSetType(FETICS ftcs, FETI feti,FETICSType type)
{
  PetscErrorCode ierr,(*func)(FETICS);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftcs,FETICS_CLASSID,1);
  PetscValidHeaderSpecific(feti,FETI_CLASSID,2);
  PetscValidCharPointer(type,3);

  ierr = PetscObjectTypeCompare((PetscObject)ftcs,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(FETICSList,type,&func);CHKERRQ(ierr);
  if (!func) SETERRQ1(PetscObjectComm((PetscObject)ftcs),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested FETICS type %s",type);
  /* Destroy the previous private FETICS context */
  if (ftcs->ops->destroy) {
    ierr               =  (*ftcs->ops->destroy)(ftcs);CHKERRQ(ierr);
    ftcs->ops->destroy = NULL;
    ftcs->data         = 0;
  }
  ierr = PetscFunctionListDestroy(&((PetscObject)ftcs)->qlist);CHKERRQ(ierr);
  /* Reinitialize function pointers in FETICSOps structure */
  ierr = PetscMemzero(ftcs->ops,sizeof(struct _FETIOps));CHKERRQ(ierr);
  /* Call the FETICSCreate_XXX routine for this particular FETI formulation */
  ftcs->feti = feti;
  ierr       = PetscObjectChangeTypeName((PetscObject)ftcs,type);CHKERRQ(ierr);
  ierr       = (*func)(ftcs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSSetFromOptions"
/*@
   FETICSSetFromOptions - Sets FETICS options from the options database.
   This routine must be called before FETICSSetUp().

   Collective on FETICS

   Input Parameter:
.  ftcs - the FETICS context

   Options Database:
.  -fetics_type: speciefies the FETICS method

   Level: begginer

.keywords: FETICS

@*/
PetscErrorCode  FETICSSetFromOptions(FETICS ftcs)
{
  PetscErrorCode ierr;
  char           type[256];
  const char*    def="csnone";
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftcs,FETICS_CLASSID,1);

  ierr = FETICSRegisterAll();CHKERRQ(ierr);
  if (!ftcs->feti) SETERRQ(PetscObjectComm((PetscObject)ftcs),PETSC_ERR_ARG_WRONGSTATE,"Error FETI context not defined");
  ierr = PetscObjectOptionsBegin((PetscObject)ftcs);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-fetics_type","FETICS","FETICSSetType",FETICSList,def,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = FETICSSetType(ftcs,ftcs->feti,type);CHKERRQ(ierr);
  } else if (!((PetscObject)ftcs)->type_name) {
    ierr = FETICSSetType(ftcs,ftcs->feti,def);CHKERRQ(ierr);
  }
  
  if (ftcs->ops->setfromoptions) {
    ierr = (*ftcs->ops->setfromoptions)(PetscOptionsObject,ftcs);CHKERRQ(ierr);
  }

  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)ftcs);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSDestroy"
/*@
   FETICSDestroy - Destroys FETICS context that was created with FETICSCreate().

   Collective on FETICS

   Input Parameter:
.  feti - the FETICS context

   Level: developer

.keywords: FETICS

.seealso: FETICSCreate(), FETICSSetUp()
@*/
PetscErrorCode FETICSDestroy(FETICS *_ftcs)
{
  PetscErrorCode ierr;
  FETICS         ftcs;

  PetscFunctionBegin;
  PetscValidPointer(_ftcs,1);
  ftcs = *_ftcs; *_ftcs = NULL;
  if (!ftcs) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ftcs,FETICS_CLASSID,1);
  if (--((PetscObject)ftcs)->refct > 0) PetscFunctionReturn(0);

  if (ftcs->ops->destroy) {ierr = (*ftcs->ops->destroy)(ftcs);CHKERRQ(ierr);}  

  ierr = PetscHeaderDestroy(&ftcs);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSSetUp"
/*@
   FETICSSetUp - Prepares the structures needed by FETICS.

   Collective on FETICS

   Input Parameter:
.  feti - the FETICS context

   Level: developer

.keywords: FETICS

.seealso: FETICSCreate(), FETICSDestroy()
@*/
PetscErrorCode  FETICSSetUp(FETICS ftcs)
{
  PetscErrorCode   ierr;
  const char*      def = "csnone";

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftcs,FETICS_CLASSID,1);
  if (ftcs->setupcalled) PetscFunctionReturn(0);
  if (!ftcs->feti) SETERRQ(PetscObjectComm((PetscObject)ftcs),PETSC_ERR_ARG_WRONGSTATE,"Error FETI context not defined");
  if (!ftcs->setupcalled) { ierr = PetscInfo(ftcs,"Setting up FETICS for first time\n");CHKERRQ(ierr);} 
  if (!((PetscObject)ftcs)->type_name) { ierr = FETICSSetType(ftcs,ftcs->feti,def);CHKERRQ(ierr);}

  if (ftcs->ops->setup) {
    ierr = (*ftcs->ops->setup)(ftcs);CHKERRQ(ierr);
  }
  if (!ftcs->setupcalled) ftcs->setupcalled++;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETICSComputeCoarseBasis"
/*@
   FETICSComputeCoarseBasis - Computes the coarse basis matrix at the interface dofs of the subdomain.  

   Input: 
.  ftcs - the FETICS context

   Output:
.  G    - coarse basis matrix at the interface computed as B*R
.  R    - coarse basis matrix for boundary and internal dofs (it can be NULL if you don't need it)

   Level: basic

.keywords: FETICS

.seealso: FETICSSetUp
@*/
PetscErrorCode FETICSComputeCoarseBasis(FETICS ftcs,Mat *G,Mat *R)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftcs,FETICS_CLASSID,1);
  PetscValidHeaderSpecific(G,MAT_CLASSID,2);
  if (R) {PetscValidHeaderSpecific(R,MAT_CLASSID,3);}
  ierr = FETICSSetUp(ftcs);CHKERRQ(ierr);
  if (ftcs->ops->computecoarsebasis) {
    ierr = (*ftcs->ops->computecoarsebasis)(ftcs,G,R);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)ftcs),PETSC_ERR_ARG_WRONGSTATE,"Error: FETICSComputeCoarseBasisI of specific FETICS method not found.");
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETICSSetFETI"
/*@
   FETICSSetFETI - Sets the FETI context which will make use of the current feti coarse space.

   Input: 
.  ftcs - the FETICS context
.  feti - the FETI context

   Level: basic

.keywords: FETICS, FETI

.seealso: FETICSSetUp, FETICSSetType
@*/
PetscErrorCode FETICSSetFETI(FETICS ftcs,FETI feti)
{  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ftcs,FETICS_CLASSID,1);
  PetscValidHeaderSpecific(feti,FETI_CLASSID,2);
  ftcs->feti = feti;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSCreate"
/*@
   FETICSCreate - Creates a FETICS context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  ftcs - location to put the FETICS context

   Options Database:
.  -fetics_type <type> - Sets the FETICS type

   Level: developer

.keywords: FETICS

.seealso: FETICSSetUp(), FETICSDestroy()
@*/
PetscErrorCode  FETICSCreate(MPI_Comm comm,FETICS *newftcs)
{
  FETICS         ftcs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newftcs,1);
  *newftcs = 0;
  ierr = FETIInitializePackage();CHKERRQ(ierr);
 
  ierr = PetscHeaderCreate(ftcs,FETICS_CLASSID,"FETICS","FETICS","FETICS",comm,FETICSDestroy,NULL);CHKERRQ(ierr);
  ftcs->setupcalled = 0;
  
  *newftcs = ftcs;
  PetscFunctionReturn(0);

}
