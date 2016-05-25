#include <private/einsfetiimpl.h>
#include <private/einsmatimpl.h>
#include <einssys.h>

PetscClassId      FETI_CLASSID;
PetscLogEvent     FETI_SetUp;
PetscBool         FETIRegisterAllCalled   = PETSC_FALSE;
PetscFunctionList FETIList                = 0;
PetscFunctionList FETIScalingList         = 0;
static PetscBool  FETIPackageInitialized  = PETSC_FALSE;


PETSC_EXTERN PetscErrorCode FETICreate_FETI1(FETI);
PETSC_EXTERN PetscErrorCode FETICreate_FETI2(FETI);
/* scaling stuff */
PETSC_EXTERN PetscErrorCode FETIScalingSetUp_none(FETI);
PETSC_EXTERN PetscErrorCode FETIScalingSetUp_rho(FETI);
PETSC_EXTERN PetscErrorCode FETIScalingSetUp_multiplicity(FETI);
PETSC_EXTERN PetscErrorCode FETIScalingDestroy(FETI);

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
  ierr = FETIRegister(FETI2,FETICreate_FETI2);CHKERRQ(ierr);
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
  feti->state = FETI_STATE_INITIAL;

  ierr = PetscObjectChangeTypeName((PetscObject)feti,type);CHKERRQ(ierr);
  ierr = (*func)(feti);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetInterfaceSolver"
/*@C
   FETISetInterfaceSolver - Sets the KSP and PC to be used for solving the FETI interface problem.

   Input Parameter:
.  ft - the FETI context.
.  kt - the KSP solver type.
.  pt - the PC type.

  Level: basic

.seealso: FETIType, FETICreate()

@*/
PetscErrorCode FETISetInterfaceSolver(FETI ft,KSPType kt,PCType pt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidCharPointer(kt,2);
  PetscValidCharPointer(pt,3);
  ft->ksp_type_interface = kt;
  ft->pc_type_interface = pt;
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
  *type = ((PetscObject)feti)->type_name;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIBuildInterfaceKSP"
/*@
   FETIBuildInterfaceKSP - Builds the KSP and PC contexts for the interface problem.

   Input Parameter:
.  feti            - the FETI context

   Level: intermediate

.keywords: FETI

@*/
PetscErrorCode FETIBuildInterfaceKSP(FETI ft)
{
  PetscErrorCode   ierr;
  MPI_Comm         comm;
  PC               pc;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr  = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  if(!ft->F) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: Matrix F for the interface problem must be first defined");
  if(!ft->ksp_type_interface) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: KSP type for the interface problem must be first defined");
  if(!ft->pc_type_interface) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: PC type for the interface problem must be first defined");
  ierr = PetscObjectIncrementTabLevel((PetscObject)ft->ksp_interface,(PetscObject)ft,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ft,(PetscObject)ft->ksp_interface);CHKERRQ(ierr);
  ierr = KSPSetType(ft->ksp_interface,ft->ksp_type_interface);CHKERRQ(ierr);
  ierr = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,ft->pc_type_interface);CHKERRQ(ierr);
  ierr = KSPSetOperators(ft->ksp_interface,ft->F,ft->F);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ft->ksp_interface,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSetNormType(ft->ksp_interface, KSP_NORM_UNPRECONDITIONED);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ft->ksp_interface,"feti_interface_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ft->ksp_interface);CHKERRQ(ierr);
  ierr = KSPSetUp(ft->ksp_interface);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICreateFMat"
/*@
   FETICreateFETIMatContext - Creates the matrix F for FETI interface problem (it is implemented as a MatShell).

   Input Parameter:
+  feti            - the FETI context
.  FETIMatMult     - pointer to function for performing the matrix vector product
.  FETIDestroyMatF - pointer to function for performing the destruction of the matrix context
-  FETIMatGetVecs  - pointer to function for getting vector(s) compatible with the matrix

   Level: intermediate

.keywords: FETI

@*/
PetscErrorCode FETICreateFMat(FETI ft,void (*FETIMatMult)(void),void (*FETIDestroyMatF)(void),void (*FETIMatGetVecs)(void))
{
  PetscErrorCode   ierr;
  FETIMat_ctx      matctx;
  MPI_Comm         comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr  = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  /* creating the mat context for the MatShell*/
  ierr = PetscNew(&matctx);CHKERRQ(ierr);
  matctx->ft = ft;
  /* creating the MatShell */
  ierr = MatCreateShellUnAsm(comm,ft->n_lambda_local,ft->n_lambda_local,ft->n_lambda,ft->n_lambda,matctx,&ft->F);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)ft->F,(PetscObject)ft,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)ft,(PetscObject)ft->F);CHKERRQ(ierr);
  ierr = MatShellUnAsmSetOperation(ft->F,MATOP_MULT,FETIMatMult);CHKERRQ(ierr);
  ierr = MatShellUnAsmSetOperation(ft->F,MATOP_DESTROY,FETIDestroyMatF);CHKERRQ(ierr);
  ierr = MatShellUnAsmSetOperation(ft->F,MATOP_GET_VECS,FETIMatGetVecs);CHKERRQ(ierr);
  ierr = MatSetUp(ft->F);CHKERRQ(ierr);
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
.   -feti_interface_<ksp_option>: options for the KSP for the interface problem
.   -feti_resetup_pc_interface: if set, PCSetUp of the PC for interface problem is called everytime that the local problem is factorized

   Level: begginer

.keywords: FETI

@*/
PetscErrorCode  FETISetFromOptions(FETI feti)
{
  PetscErrorCode ierr;
  char           type[256];
  const char*    def="feti1";
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

  ierr = PetscOptionsBool("-feti_resetup_pc_interface","If set, PCSetUp of the PC for interface problem is called everytime that the local problem is factorized",
			  "none",feti->resetup_pc_interface,&feti->resetup_pc_interface,NULL);CHKERRQ(ierr);
  
  if (feti->ops->setfromoptions) {
    ierr = (*feti->ops->setfromoptions)(PetscOptionsObject,feti);CHKERRQ(ierr);
  }

  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)feti);CHKERRQ(ierr);
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
  feti->state = FETI_STATE_INITIAL;
  /* destroying FETI objects */
  ierr = MatDestroy(&feti->F);CHKERRQ(ierr);
  ierr = KSPDestroy(&feti->ksp_neumann);CHKERRQ(ierr);
  ierr = KSPDestroy(&feti->ksp_interface);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->B_delta);CHKERRQ(ierr);
  ierr = MatDestroy(&feti->B_Ddelta);CHKERRQ(ierr);
  ierr = VecDestroy(&feti->d);CHKERRQ(ierr);
  ierr = VecDestroy(&feti->multiplicity);CHKERRQ(ierr);
  ierr = VecDestroy(&feti->lambda_local);CHKERRQ(ierr);
  ierr = VecDestroy(&feti->lambda_global);CHKERRQ(ierr);
  ierr = FETIScalingDestroy(feti);CHKERRQ(ierr);
  ierr = VecExchangeDestroy(&feti->exchange_lambda);CHKERRQ(ierr);
  if (feti->n_neigh_lb > -1) {
    ierr = ISLocalToGlobalMappingRestoreInfo(feti->mapping_lambda,&(feti->n_neigh_lb),&(feti->neigh_lb),
					     &(feti->n_shared_lb),&(feti->shared_lb));CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&feti->mapping_lambda);CHKERRQ(ierr);
  ierr = SubdomainDestroy(&feti->subdomain);CHKERRQ(ierr);

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
  ierr = PetscFunctionListDestroy(&FETIScalingList);CHKERRQ(ierr);
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
  if (feti->state>=FETI_STATE_SETUP_END) PetscFunctionReturn(0);
  if (!feti->subdomain) SETERRQ(PetscObjectComm((PetscObject)feti),PETSC_ERR_ARG_WRONGSTATE,"Error Subdomain not defined");
  ierr = SubdomainCheckState(feti->subdomain);CHKERRQ(ierr);
  
  if (feti->state == FETI_STATE_INITIAL) { ierr = PetscInfo(feti,"Setting up FETI for first time\n");CHKERRQ(ierr);} 
  if (!((PetscObject)feti)->type_name) { ierr = FETISetType(feti,def);CHKERRQ(ierr);}

  ierr = PetscLogEventBegin(FETI_SetUp,feti,0,0,0);CHKERRQ(ierr);
  ierr = SubdomainSetUp(feti->subdomain,(PetscBool)(feti->state>=FETI_STATE_SETUP_INI));CHKERRQ(ierr);

  if (feti->ops->setup) {
    ierr = (*feti->ops->setup)(feti);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(FETI_SetUp,feti,0,0,0);CHKERRQ(ierr);
  if (feti->state == FETI_STATE_INITIAL) feti->state = FETI_STATE_SETUP_END;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETIGetKSPInterface"
/*@
   FETIGetKSPInterface - Gets the KSP created for solving the interface problem. 

   Input: 
.  ft - the FETI context

   Output:
.  ksp_interface  - the KSP for the interface problem

   Level: basic

.keywords: FETI

.seealso: FETISetUp
@*/
PetscErrorCode FETIGetKSPInterface(FETI ft,KSP *ksp_interface)
{
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  if (!ft->ksp_interface) SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: ksp_interface has not been yet created.");
  *ksp_interface = ft->ksp_interface;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISolve"
/*@
   FETISolve - Computes the primal solution by using the FETI method.

   Input: 
.  ft - the FETI context

   Output:
.  u  - vector to store the solution

   Level: basic

.keywords: FETI

.seealso: FETISetUp
@*/
PetscErrorCode FETISolve(FETI ft,Vec u)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(u,VEC_CLASSID,2);
  if (ft->state==FETI_STATE_SOLVED) PetscFunctionReturn(0);
  ierr = FETISetUp(ft);CHKERRQ(ierr);
  if (ft->ops->computesolution) {
    ierr = PetscObjectTypeCompare((PetscObject)u,VECMPIUNASM,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = (*ft->ops->computesolution)(ft,u);CHKERRQ(ierr);
    } else {
      Vec u_local;
      ierr = VecUnAsmGetLocalVector(u,&u_local);CHKERRQ(ierr);
      ierr = (*ft->ops->computesolution)(ft,u_local);CHKERRQ(ierr);
      ierr = VecUnAsmRestoreLocalVector(u,u_local);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: Compute Solution of specific FETI method not found.");
  }
  ft->state = FETI_STATE_SOLVED;
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
  if (ft->state == FETI_STATE_SOLVED) {ft->state = FETI_STATE_SETUP_INI;};
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISetMat"
/*@
   FETISetMat - Sets the local system matrix for the current process by providing a locally unassembled matrix.

   Input Parameter:
.  ft    - The FETI context
.  mat   - The matrix

   Level: beginner

.keywords: FETI, local system matrix

.seealso: FETISetLocalRHS(), FETISetMapping()
@*/
PetscErrorCode FETISetMat(FETI ft,Mat mat)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  LGMat_ctx      mat_ctx;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSHELLUNASM,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot set non-MATSHELLUNASM matrix");
  ierr = MatShellUnAsmGetContext(mat,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = SubdomainSetLocalMat(ft->subdomain,mat_ctx->localA);CHKERRQ(ierr);
  if (ft->state == FETI_STATE_SOLVED) {ft->state = FETI_STATE_SETUP_INI;};
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISetRHS"
/*@
   FETISetRHS - Sets the local system RHS for the current process by providing a globally unassembled vector.

   Input Parameter:
.  ft  - The FETI context
.  rhs - The system rhs

   Level: beginner

.keywords: FETI, local system rhs

.seealso: FETISetLocalMat(), FETISetMapping()
@*/
PetscErrorCode FETISetRHS(FETI ft,Vec rhs)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  Vec            rhs_local;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(rhs,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)rhs,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)rhs),PETSC_ERR_SUP,"Cannot set non-VECMPIUNASM vector");
  ierr = VecUnAsmGetLocalVectorRead(rhs,&rhs_local);CHKERRQ(ierr);
  ierr = SubdomainSetLocalRHS(ft->subdomain,rhs_local);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(rhs,rhs_local);CHKERRQ(ierr);
  if (ft->state == FETI_STATE_SOLVED) {ft->state = FETI_STATE_SETUP_INI;};
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
  if (ft->state == FETI_STATE_SOLVED) {ft->state = FETI_STATE_SETUP_INI;};
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETISetMappingAndGlobalSize"
/*@
   FETISetMappingAndGlobalSize - Sets the mapping from local to global numbering of 
   DOFs, and the global number of DOFs.

   Input Parameter:
.  ft    - The FETI context
.  isg2l - A mapping from local to global numering of DOFs
.  N     - Global number of DOFs

   Level: beginner

.keywords: FETI, local to global numbering of DOFs

.seealso: FETISetLocalRHS(), FETISetLocalMat()
@*/
PetscErrorCode FETISetMappingAndGlobalSize(FETI ft,ISLocalToGlobalMapping isg2l,PetscInt N)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  PetscValidHeaderSpecific(isg2l,IS_LTOGM_CLASSID,2);
  ierr = SubdomainSetMapping(ft->subdomain,isg2l);CHKERRQ(ierr);
  ierr = SubdomainSetGlobalSize(ft->subdomain,N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetReSetupPCInterface"
/*@C 
   FETISetReSetupPCInterface - Sets the value of the flag
   controlling the call to PCSetUp of the PC corresponding to
   interface problem.

   Input Parameters:
+  ft                    - the FETI context 
-  resetup_pc_interface  - boolean value to set

   Level: beginner

.keywords: FETI

@*/
PETSC_EXTERN PetscErrorCode FETISetReSetupPCInterface(FETI ft,PetscBool resetup_pc_interface)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ft->resetup_pc_interface = resetup_pc_interface;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIComputeForceNormLocal"
/*@C
   FETIComputeForceNormLocal - Computes the local norm of a vector as ||res_s - B_s^T lambda||_{norm_type}

   Input Parameters:
+  ft      - the FETI context 
.  vec     - the vector from which to compute the norm
.  type    - norm type: NORM_1, NORM_2, NORM_INFINITY

   Output Parameters:
+  norm    - the computed norm 

   Level: beginner

.keywords: FETI

@*/
PETSC_EXTERN PetscErrorCode FETIComputeForceNormLocal(FETI ft,Vec vec,NormType type,PetscReal *norm) {
  PetscErrorCode    ierr;
  Subdomain         sd;
  Vec               lambda_local,vlocal;
  PetscBool         flg;
  
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)vec,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if (!flg) {
    vlocal = vec;
  } else {
    ierr = VecUnAsmGetLocalVectorRead(vec,&vlocal);CHKERRQ(ierr);
  }    
  if (ft->lambda_global) {
    sd = ft->subdomain;
    /* computing B_delta^T*lambda */
    ierr = VecUnAsmGetLocalVector(ft->lambda_global,&lambda_local);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft->B_delta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
    ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    /* computing vec - B_delta^T*lambda */
    ierr = VecAYPX(sd->vec1_N,-1.0,vlocal);CHKERRQ(ierr);
    /* compute the norm */
    ierr = VecNorm(sd->vec1_N,type,norm);CHKERRQ(ierr);
  } else {
    /* compute the norm of vec if no lambda is available*/
    ierr = VecNorm(vlocal,type,norm);CHKERRQ(ierr);
  }
  if (flg) {
    ierr = VecUnAsmRestoreLocalVectorRead(vec,vlocal);CHKERRQ(ierr);
  }    

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIComputeForceNorm"
/*@C
   FETIComputeForceNorm - Computes the norm of a vector as sum_s(||res_s - B_s^T lambda||_{norm_type})

   Input Parameters:
+  ft      - the FETI context 
.  vec     - the vector from which to compute the norm
.  type    - norm type: NORM_1, NORM_2, NORM_INFINITY

   Output Parameters:
+  norm    - the computed norm 

   Level: beginner

.keywords: FETI

@*/
PETSC_EXTERN PetscErrorCode FETIComputeForceNorm(FETI ft,Vec vec,NormType type,PetscReal *norm) {
  PetscErrorCode    ierr;
  Subdomain         sd;
  Vec               lambda_local,vlocal;
  PetscBool         flg;
  PetscReal         lnorm;
  
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)vec,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if (!flg) {
    vlocal = vec;
  } else {
    ierr = VecUnAsmGetLocalVectorRead(vec,&vlocal);CHKERRQ(ierr);
  }    
  if (ft->lambda_global) {
    sd = ft->subdomain;
    /* computing B_delta^T*lambda */
    ierr = VecUnAsmGetLocalVector(ft->lambda_global,&lambda_local);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft->B_delta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
    ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    /* computing vec - B_delta^T*lambda */
    ierr = VecAYPX(sd->vec1_N,-1.0,vlocal);CHKERRQ(ierr);
    /* compute the norm */
    ierr = VecNorm(sd->vec1_N,type,&lnorm);CHKERRQ(ierr);
  } else {
    /* compute the norm of vec if no lambda is available*/
    ierr = VecNorm(vlocal,type,&lnorm);CHKERRQ(ierr);
  }
  ierr = MPI_Allreduce(&lnorm,norm,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)vec));CHKERRQ(ierr);
  if (flg) {
    ierr = VecUnAsmRestoreLocalVectorRead(vec,vlocal);CHKERRQ(ierr);
  }    
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIBuildLambdaAndB"
/*@
   FETIBuildLambdaAndB - Computes the B operator and the vector lambda of 
   the interface problem.

   Input Parameters:
.  ft - the FETI context

   Notes: 
   In a future this rutine could be moved to the FETI class.

   Level: developer
   
@*/
PetscErrorCode FETIBuildLambdaAndB(FETI ft)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  IS                subset,subset_mult,subset_n;
  PetscBool         fully_redundant;
  PetscInt          i,j,s,n_boundary_dofs,n_global_lambda,partial_sum,up;
  PetscInt          cum,n_lambda_local,n_lambda_for_dof,dual_size,n_neg_values,n_pos_values;
  PetscMPIInt       rank;
  PetscInt          *dual_dofs_boundary_indices,*aux_local_numbering_1;
  const PetscInt    *aux_global_numbering,*indices;
  PetscInt          *aux_sums,*cols_B_delta,*l2g_indices;
  PetscScalar       *array,*vals_B_delta,*vals_B_Ddelta;
  PetscInt          *aux_local_numbering_2;
  PetscScalar       scalar_value;
  Subdomain         sd = ft->subdomain;
  const PetscScalar *Warray;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  if(!ft->Wscaling) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: FETIScalingSetUp must be first called");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* Default type of lagrange multipliers is non-redundant */
  fully_redundant = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-feti_fullyredundant",&fully_redundant,NULL);CHKERRQ(ierr);

  /* Evaluate local and global number of lagrange multipliers */
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  n_lambda_local = 0;
  partial_sum = 0;
  n_boundary_dofs = 0;
  dual_size = sd->n_B;
  ierr = PetscMalloc1(dual_size,&dual_dofs_boundary_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscMalloc1(dual_size,&aux_local_numbering_2);CHKERRQ(ierr);
  
  ierr = ISGetIndices(sd->is_B_local,&indices);CHKERRQ(ierr); 
  ierr = VecGetArray(sd->vec1_B,&array);CHKERRQ(ierr);
  for (i=0;i<dual_size;i++){
    j = sd->count[i]; /* RECALL: sd->count[i] does not count myself */
    n_boundary_dofs++;
    if (fully_redundant) {
      /* fully redundant set of lagrange multipliers */
      n_lambda_for_dof = (j*(j+1))/2;
    } else {
      n_lambda_for_dof = j;
    }
    n_lambda_local += j;
    /* needed to evaluate global number of lagrange multipliers */
    array[i]=(1.0*n_lambda_for_dof)/(j+1.0); /* already scaled for the next global sum */
    /* store some data needed */
    dual_dofs_boundary_indices[partial_sum] = n_boundary_dofs-1;
    aux_local_numbering_1[partial_sum] = indices[i];
    aux_local_numbering_2[partial_sum] = n_lambda_for_dof;
    partial_sum++;
  }
  ierr = VecRestoreArray(sd->vec1_B,&array);CHKERRQ(ierr);
  ierr = ISRestoreIndices(sd->is_B_local,&indices);CHKERRQ(ierr);
  ft->n_lambda_local = n_lambda_local;
  
  /* compute ft->n_lambda */
  ierr = VecSet(sd->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterUABegin(sd->N_to_B,sd->vec1_B,sd->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterUAEnd(sd->N_to_B,sd->vec1_B,sd->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecExchangeBegin(sd->exchange_vec1global,sd->vec1_global,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(sd->exchange_vec1global,sd->vec1_global,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecUnAsmSum(sd->vec1_global,&scalar_value);CHKERRQ(ierr);
  ft->n_lambda = (PetscInt)PetscRealPart(scalar_value);
  
  /* compute global ordering of lagrange multipliers and associate l2g map */
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_1,PETSC_COPY_VALUES,&subset_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(sd->mapping,subset_n,&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = PetscFree(aux_local_numbering_1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,partial_sum,aux_local_numbering_2,PETSC_OWN_POINTER,&subset_mult);CHKERRQ(ierr);
  ierr = ISSubsetNumbering(subset,subset_mult,&i,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);
  if (i != ft->n_lambda) {
    SETERRQ3(comm,PETSC_ERR_PLIB,"Error in %s: global number of multipliers mismatch! (%d!=%d)\n",__FUNCT__,ft->n_lambda,i);
  }
  /* Compute B_delta (local actions) */
  ierr = PetscMalloc1(sd->n_neigh,&aux_sums);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&l2g_indices);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&vals_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&vals_B_Ddelta);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&cols_B_delta);CHKERRQ(ierr);
  ierr = ISGetIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ft->Wscaling,&Warray);CHKERRQ(ierr);
  n_global_lambda=0;
  partial_sum=0;
  cum = 0;
  for (i=0;i<dual_size;i++) {
    n_global_lambda = aux_global_numbering[cum];
    j = sd->count[i]; /* "sd->count[aux_local_numbering_1[i]]": aux_local_numbering_1[i] primal dof number of the boundary */
    aux_sums[0]=0;
    for (s=1;s<j;s++) {
      aux_sums[s]=aux_sums[s-1]+j-s+1;
    }
    n_neg_values = 0;
    
    while(n_neg_values < j && sd->neighbours_set[i][n_neg_values] < rank){
      n_neg_values++;
    }
    
    n_pos_values = j - n_neg_values;
    if (fully_redundant) {
      for (s=0;s<n_neg_values;s++) {
	up = dual_dofs_boundary_indices[i];
        l2g_indices    [partial_sum+s]=aux_sums[s]+n_neg_values-s-1+n_global_lambda;
        cols_B_delta   [partial_sum+s]=up;
        vals_B_delta   [partial_sum+s]=-1.0;
	vals_B_Ddelta  [partial_sum+s]=-Warray[up];
      }
      for (s=0;s<n_pos_values;s++) {
	up = dual_dofs_boundary_indices[i];
        l2g_indices    [partial_sum+s+n_neg_values]=aux_sums[n_neg_values]+s+n_global_lambda;
        cols_B_delta   [partial_sum+s+n_neg_values]=up;
        vals_B_delta   [partial_sum+s+n_neg_values]=1.0;
	vals_B_Ddelta  [partial_sum+s+n_neg_values]=Warray[up];	
      }
      partial_sum += j;
    } else {
      /* l2g_indices and default cols and vals of B_delta */
      up = dual_dofs_boundary_indices[i];
      for (s=0;s<j;s++) {
        l2g_indices    [partial_sum+s]=n_global_lambda+s;
        cols_B_delta   [partial_sum+s]=up;
        vals_B_delta   [partial_sum+s]=0.0;
	vals_B_Ddelta  [partial_sum+s]=0.0;	
      }
      /* B_delta */
      if ( n_neg_values > 0 ) { /* there's a rank next to me to the left */
        vals_B_delta   [partial_sum+n_neg_values-1]=-1.0;
	vals_B_Ddelta  [partial_sum+n_neg_values-1]=-Warray[up];	
      }
      if ( n_neg_values < j ) { /* there's a rank next to me to the right */
        vals_B_delta   [partial_sum+n_neg_values]=1.0;
	vals_B_Ddelta  [partial_sum+n_neg_values]=Warray[up];
      }
      partial_sum += j;
    }
    cum += aux_local_numbering_2[i];
  }
  ierr = VecRestoreArrayRead(ft->Wscaling,&Warray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(subset_n,&aux_global_numbering);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_mult);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = PetscFree(aux_sums);CHKERRQ(ierr);
  ierr = PetscFree(dual_dofs_boundary_indices);CHKERRQ(ierr);

  /* Create global_lambda */
  ierr = VecCreate(comm,&ft->lambda_global);CHKERRQ(ierr);
  ierr = VecSetSizes(ft->lambda_global,n_lambda_local,ft->n_lambda);CHKERRQ(ierr);
  ierr = VecSetType(ft->lambda_global,VECMPIUNASM);CHKERRQ(ierr);
  /* create local to global mapping and neighboring information for lambda */
  ierr = ISLocalToGlobalMappingCreate(comm,1,n_lambda_local,l2g_indices,PETSC_COPY_VALUES,&ft->mapping_lambda);
  ierr = ISLocalToGlobalMappingGetInfo(ft->mapping_lambda,&(ft->n_neigh_lb),&(ft->neigh_lb),&(ft->n_shared_lb),&(ft->shared_lb));CHKERRQ(ierr);
  ierr = VecExchangeCreate(ft->lambda_global,ft->n_neigh_lb,ft->neigh_lb,ft->n_shared_lb,ft->shared_lb,PETSC_USE_POINTER,&ft->exchange_lambda);CHKERRQ(ierr);
  /* set multiplicity for lambdas */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n_lambda_local,&ft->multiplicity);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_lambda_local,&aux_local_numbering_2);CHKERRQ(ierr);
  ierr = PetscCalloc1(n_lambda_local,&array);CHKERRQ(ierr);
  for (i=0;i<ft->n_lambda_local;i++) aux_local_numbering_2[i] = i;
  for (i=0;i<ft->n_neigh_lb;i++) 
    for (j=0;j<ft->n_shared_lb[i];j++)
      array[ft->shared_lb[i][j]] += 1;

  ierr = VecSetValues(ft->multiplicity,n_lambda_local,aux_local_numbering_2,array,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(ft->multiplicity);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ft->multiplicity);CHKERRQ(ierr);
  ierr = PetscFree(array);CHKERRQ(ierr);
  ierr = PetscFree(aux_local_numbering_2);CHKERRQ(ierr);
  ierr = VecUnAsmSetMultiplicity(ft->lambda_global,ft->multiplicity);CHKERRQ(ierr);
  /* Create local part of B_delta */
  ierr = MatCreate(PETSC_COMM_SELF,&ft->B_delta);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&ft->B_Ddelta);CHKERRQ(ierr);
  ierr = MatSetSizes(ft->B_delta,n_lambda_local,sd->n_B,n_lambda_local,sd->n_B);CHKERRQ(ierr);
  ierr = MatSetSizes(ft->B_Ddelta,n_lambda_local,sd->n_B,n_lambda_local,sd->n_B);CHKERRQ(ierr);
  ierr = MatSetType(ft->B_delta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetType(ft->B_Ddelta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(ft->B_delta,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(ft->B_Ddelta,1,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(ft->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(ft->B_Ddelta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<n_lambda_local;i++) {
    ierr = MatSetValue(ft->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(ft->B_Ddelta,i,cols_B_delta[i],vals_B_Ddelta[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(ft->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (ft->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ft->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (ft->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(vals_B_delta);CHKERRQ(ierr);
  ierr = PetscFree(vals_B_Ddelta);CHKERRQ(ierr);
  ierr = PetscFree(cols_B_delta);CHKERRQ(ierr);
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

   Options Database:
.  -feti_type <type> - Sets the FETI type
.  -feti_interface_<ksp_option> - options for the KSP for the interface problem
.  -feti_scaling_type - Sets the scaling type
.  -feti_scaling_factor - Sets a scaling factor different from one

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

  /* adding scaling types*/
  ierr = PetscFunctionListAdd(&FETIScalingList,SCNONE,FETIScalingSetUp_none);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&FETIScalingList,SCRHO,FETIScalingSetUp_rho);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&FETIScalingList,SCMULTIPLICITY,FETIScalingSetUp_multiplicity);CHKERRQ(ierr);

  feti->state                = FETI_STATE_INITIAL;
  feti->setfromoptionscalled = 0;
  feti->multiplicity         = 0;
  feti->data                 = 0;
  feti->lambda_local         = 0;
  feti->lambda_global        = 0;
  feti->mapping_lambda       = 0;
  feti->n_lambda_local       = 0;
  feti->exchange_lambda      = 0;
  feti->n_lambda             = -1;
  feti->F                    = 0;
  feti->d                    = 0;
  feti->ksp_type_interface   = 0;
  feti->pc_type_interface    = 0;
  feti->B_delta              = 0;
  feti->B_Ddelta             = 0;
  feti->ksp_neumann          = 0;
  feti->n_neigh_lb           = -1;
  feti->neigh_lb             = 0;
  feti->n_shared_lb          = 0;
  feti->shared_lb            = 0;
  feti->mat_state            = -1;
  feti->resetup_pc_interface = PETSC_TRUE;
  /* scaling variables initialization*/
  feti->Wscaling             = 0;
  feti->scaling_factor       = 1.;
  feti->scaling_type         = SCUNK;

  ierr = KSPCreate(comm,&feti->ksp_interface);CHKERRQ(ierr);
  
  *newfeti = feti;
  PetscFunctionReturn(0);

}
