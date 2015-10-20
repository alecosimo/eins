#include <private/einsfetiimpl.h>

PetscClassId      FETI_CLASSID;
PetscLogEvent     FETI_SetUp;
PetscBool         FETIRegisterAllCalled   = PETSC_FALSE;
PetscFunctionList FETIList                = 0;
static PetscBool  FETIPackageInitialized  = PETSC_FALSE;

PETSC_EXTERN PetscErrorCode FETICreate_FETI1(FETI);

#undef __FUNCT__
#define __FUNCT__ "FETIRegister"
/*@C
  FETIRegister -  Adds a FETI method.

   Not collective

   Input Parameters:
+  name_feti - name of a new user-defined FETI method
-  routine_create - routine to create method context

   Level: advanced

.keywords: FETI

.seealso: FETIRegisterAll()
@*/
PetscErrorCode  FETIRegister(const char sname[],PetscErrorCode (*function)(FETI))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&FETIList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIRegisterAll"
/*@C
   FETIRegisterAll - Registers all of the FETI methods in the FETI package.

   Not Collective

   Level: advanced

.keywords: FETI

.seealso: FETIRegister()
@*/
PetscErrorCode  FETIRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (FETIRegisterAllCalled) PetscFunctionReturn(0);
  FETIRegisterAllCalled = PETSC_TRUE;

  ierr = FETIRegister(FETI1,FETICreate_FETI1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetType"
/*@C
   FETISetType - Builds FETI for the particular FETI type

   Collective on FETI

   Input Parameter:
+  feti - the FETI context.
-  type - a FETI formulation

   Options Database Key:
.  -feti_type <type> - Sets FETI type

  Level: intermediate

.seealso: FETIType, FETIRegister(), FETICreate()

@*/
PetscErrorCode  FETISetType(FETI feti,FETIType type)
{
  PetscErrorCode ierr,(*func)(FETI);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(feti,FETI_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)feti,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(FETIList,type,&func);CHKERRQ(ierr);
  if (!func) SETERRQ1(PetscObjectComm((PetscObject)feti),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested FETI type %s",type);
  /* Destroy the previous private FETI context */
  if (feti->ops->destroy) {
    ierr               =  (*feti->ops->destroy)(feti);CHKERRQ(ierr);
    feti->ops->destroy = NULL;
    feti->data         = 0;
  }
  ierr = PetscFunctionListDestroy(&((PetscObject)feti)->qlist);CHKERRQ(ierr);
  /* Reinitialize function pointers in FETIOps structure */
  ierr = PetscMemzero(feti->ops,sizeof(struct _FETIOps));CHKERRQ(ierr);
  /* Call the FETICreate_XXX routine for this particular FETI formulation */
  feti->setupcalled = 0;

  ierr = PetscObjectChangeTypeName((PetscObject)feti,type);CHKERRQ(ierr);
  ierr = (*func)(feti);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIGetType"
/*@C
   FETIGetType - Gets the FETI type and name (as a string) the FETI context.

   Not Collective

   Input Parameter:
.  feti - the FETI context

   Output Parameter:
.  type - name of FETI method

   Level: intermediate

.keywords: FETI

.seealso: FETISetType()

@*/
PetscErrorCode FETIGetType(FETI feti,FETIType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(feti,FETI_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)feti)->type_name;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetFromOptions"
/*@
   FETISetFromOptions - Sets FETI options from the options database.
   This routine must be called before FETISetUp().

   Collective on FETI

   Input Parameter:
.  feti - the FETI context

   Options Database:
.   -feti_type: speciefies the FETI method

   Level: begginer

.keywords: FETI

@*/
PetscErrorCode  FETISetFromOptions(FETI feti)
{
  PetscErrorCode ierr;
  char           type[256];
  const char*    def="FETI1";
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(feti,FETI_CLASSID,1);

  ierr = FETIRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)feti);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-feti_type","FETI","FETISetType",FETIList,def,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = FETISetType(feti,type);CHKERRQ(ierr);
  } else if (!((PetscObject)feti)->type_name) {
    ierr = FETISetType(feti,def);CHKERRQ(ierr);
  }

  if (feti->ops->setfromoptions) {
    ierr = (*feti->ops->setfromoptions)(PetscOptionsObject,feti);CHKERRQ(ierr);
  }

  ierr = PetscObjectProcessOptionsHandlers((PetscObject)feti);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  feti->setfromoptionscalled++;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIDestroy"
/*@
   FETIDestroy - Destroys FETI context that was created with FETICreate().

   Collective on FETI

   Input Parameter:
.  feti - the FETI context

   Level: developer

.keywords: FETI

.seealso: FETICreate(), FETISetUp()
@*/
PetscErrorCode FETIDestroy(FETI *_feti)
{
  PetscErrorCode ierr;
  FETI           feti;

  PetscFunctionBegin;
  PetscValidPointer(_feti,1);
  feti = *_feti; *_feti = NULL;
  if (!feti) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(feti,FETI_CLASSID,1);
  if (--((PetscObject)feti)->refct > 0) PetscFunctionReturn(0);
  feti->setupcalled = 0;
  /* destroying FETI objects */
  ierr = SubdomainDestroy(&feti->subdomain);CHKERRQ(ierr);
  ierr = KSPDestroy(&feti->ksp_neumann);CHKERRQ(ierr);
  ierr = KSPDestroy(&feti->ksp_interface);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->B_delta);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->B_Ddelta);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->Wscaling);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->A_II);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->A_BB);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->A_IB);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->F);CHKERRQ(ierr);
  ierr = VecDestroy(&feti->d);CHKERRQ(ierr);
  ierr = VecDestroy(&feti->lambda_local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&feti->l2g_lambda);CHKERRQ(ierr);
    
  if (feti->ops->destroy) {ierr = (*feti->ops->destroy)(feti);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(&feti);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIFinalizePackage"
/*@C
  FETIFinalizePackage - This function destroys everything in the Petsc interface related to the FETI package. It is
  called from PetscFinalize() which is called from EinsFinalize().

  Level: developer

.keywords: Petsc
.seealso: PetscFinalize()
@*/
PetscErrorCode  FETIFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&FETIList);CHKERRQ(ierr);
  FETIPackageInitialized = PETSC_FALSE;
  FETIRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIInitializePackage"
/*@C
  FETIInitializePackage - Initializes the FETI package.

  Level: developer

.keywords: FETI
@*/
PetscErrorCode  FETIInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (FETIPackageInitialized) PetscFunctionReturn(0);
  FETIPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("FETI",&FETI_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = FETIRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("FETISetUp",FETI_CLASSID,&FETI_SetUp);CHKERRQ(ierr);
  /* Set FETIFinalizePackage */
  ierr = PetscRegisterFinalize(FETIFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp"
/*@
   FETISetUp - Prepares the structures needed by the FETI solver.

   Collective on FETI

   Input Parameter:
.  feti - the FETI context

   Level: developer

.keywords: FETI

.seealso: FETICreate(), FETIDestroy()
@*/
PetscErrorCode  FETISetUp(FETI feti)
{
  PetscErrorCode   ierr;
  const char*      def = "FETI1";

  PetscFunctionBegin;
  PetscValidHeaderSpecific(feti,FETI_CLASSID,1);

  if (!feti->subdomain) SETERRQ(PetscObjectComm((PetscObject)feti),PETSC_ERR_ARG_WRONGSTATE,"Error Subdomain not defined");
  ierr = SubdomainCheckState(feti->subdomain);CHKERRQ(ierr);
  
  if (!feti->setupcalled) { ierr = PetscInfo(feti,"Setting up FETI for first time\n");CHKERRQ(ierr);} 
  if (!((PetscObject)feti)->type_name) { ierr = FETISetType(feti,def);CHKERRQ(ierr);}

  ierr = PetscLogEventBegin(FETI_SetUp,feti,0,0,0);CHKERRQ(ierr);
  ierr = SubdomainSetUp(feti->subdomain,feti->setupcalled);CHKERRQ(ierr);

  if (feti->ops->setup) {
    ierr = (*feti->ops->setup)(feti);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(FETI_SetUp,feti,0,0,0);CHKERRQ(ierr);
  if (!feti->setupcalled) feti->setupcalled = 1;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETICreateGlobalWorkingVec"
/*@
   FETICreateGlobalWorkingVec - Creates the global (distributed) working vector by duplicating a given vector.

   Input Parameter:
.  ft  - The FETI context
.  vec - The global vector to use in the duplication

   Level: intermediate

.keywords: FETI, working global vector
@*/
PetscErrorCode FETICreateGlobalWorkingVec(FETI ft,Vec vec)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  ierr = SubdomainCreateGlobalWorkingVec(ft->subdomain,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISetLocalMat"
/*@
   FETISetLocalMat - Sets the local system matrix for the current process.

   Input Parameter:
.  ft    - The FETI context
.  local_mat - The local system matrix

   Level: beginner

.keywords: FETI, local system matrix

.seealso: FETISetLocalRHS(), FETISetMapping()
@*/
PetscErrorCode FETISetLocalMat(FETI ft,Mat local_mat)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(local_mat,MAT_CLASSID,2);
  ierr = SubdomainSetLocalMat(ft->subdomain,local_mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISetLocalRHS"
/*@
   FETISetLocalRHS - Sets the local system RHS for the current process.

   Input Parameter:
.  ft  - The FETI context
.  rhs - The local system rhs

   Level: beginner

.keywords: FETI, local system rhs

.seealso: FETISetLocalMat(), FETISetMapping()
@*/
PetscErrorCode FETISetLocalRHS(FETI ft,Vec rhs)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  ierr = SubdomainSetLocalRHS(ft->subdomain,rhs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISetMapping"
/*@
   FETISetMapping - Sets the mapping from local to global numbering of DOFs

   Input Parameter:
.  ft    - The FETI context
.  isg2l - A mapping from local to global numering of DOFs

   Level: beginner

.keywords: FETI, local to global numbering of DOFs

.seealso: FETISetLocalRHS(), FETISetLocalMat()
@*/
PetscErrorCode FETISetMapping(FETI ft,ISLocalToGlobalMapping isg2l)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(isg2l,IS_LTOGM_CLASSID,2);
  ierr = SubdomainSetMapping(ft->subdomain,isg2l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICreate"
/*@
   FETICreate - Creates a FETI context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  feti - location to put the FETI context

   Options
.  -feti_type <type> - Sets the FETI type

   Level: developer

.keywords: FETI

.seealso: FETISetUp(), FETIDestroy()
@*/
PetscErrorCode  FETICreate(MPI_Comm comm,FETI *newfeti)
{
  FETI           feti;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newfeti,1);
  *newfeti = 0;
  ierr = FETIInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(feti,FETI_CLASSID,"FETI","FETI","FETI",comm,FETIDestroy,NULL);CHKERRQ(ierr);
  ierr = SubdomainCreate(((PetscObject)feti)->comm,&feti->subdomain);CHKERRQ(ierr);
  
  feti->setupcalled          = 0;
  feti->setfromoptionscalled = 0;
  feti->data                 = 0;
  feti->A_II                 = 0;
  feti->A_BB                 = 0;
  feti->A_IB                 = 0;
  feti->Wscaling             = 0;
  feti->scalingType          = 0;
  feti->lambda_local         = 0;
  feti->n_local_lambda       = 0;
  feti->l2g_lambda           = 0;
  feti->n_lambda             = -1;
  feti->F                    = 0;
  feti->d                    = 0;
  feti->ksp_interface        = 0;
  feti->ksp_type_interface   = 0;
  feti->pc_type_interface    = 0;
  feti->B_delta              = 0;
  feti->B_Ddelta             = 0;
  feti->ksp_neumann          = 0;
  
  *newfeti = feti;
  PetscFunctionReturn(0);

}
