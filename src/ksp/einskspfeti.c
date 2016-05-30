#include <../src/ksp/einskspfetiimpl.h>

static PetscErrorCode KSPSolve_FETI(KSP);
static PetscErrorCode KSPDestroy_FETI(KSP);
static PetscErrorCode KSPSetUp_FETI(KSP);


#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_FETI"
static PetscErrorCode KSPSetUp_FETI(KSP ksp)
{
  KSP_FETI       *ft = (KSP_FETI*)ksp->data;

  PetscFunctionBegin;
  if(!ft->feti) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Error: the feti context must be first defined");
  PetscFunctionReturn(0);
}

/* #undef __FUNCT__ */
/* #define __FUNCT__ "KSPView_FETI" */
/* static PetscErrorCode KSPView_FETI(KSP,PetscViewer); */
/* static PetscErrorCode KSPView_FETI(KSP ksp,PetscViewer viewer) */
/* { */
/*   KSP_FETI       *ft = (KSP_FETI*)ksp->data; */
/*   PetscErrorCode ierr; */

/*   PetscFunctionBegin; */
/*   ierr = FETIView(ft->feti);CHKERRQ(ierr); */
/*   PetscFunctionReturn(0); */
/* } */



#undef __FUNCT__
#define __FUNCT__ "KSPGetFETI"
/*@
  KSPGetFETI - Gets the FETI context defined in the KSP

  Input Parameters:
+  ksp - the Krylov space context

  Output Parameters:
+  feti - the FETI Context

  Level: basic

.seealso: KSPFETI, KSPSetFETI()
@*/
PETSC_EXTERN PetscErrorCode KSPGetFETI(KSP ksp,FETI *feti)
{
  KSP_FETI       *ft = (KSP_FETI*)ksp->data; 
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(feti,2);
  if(!ft->feti) {
    ierr = FETICreate(PetscObjectComm((PetscObject)ksp),&ft->feti);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ft->feti,(PetscObject)ksp,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ksp,(PetscObject)ft->feti);CHKERRQ(ierr);
  }
  *feti = ft->feti;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPSetFETI"
/*@
  KSPSetFETI - Sets the FETI context for the KSP

  Input Parameters:
+  ksp - the Krylov space context
-  feti - the FETI Context

  Level: basic

.seealso: KSPFETI, KSPGetFETI()
@*/
PETSC_EXTERN PetscErrorCode KSPSetFETI(KSP ksp,FETI feti)
{
  KSP_FETI         *ft = (KSP_FETI*)ksp->data;
  PetscErrorCode   ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(feti,FETI_CLASSID,2);
  PetscCheckSameComm(ksp,1,feti,2);
  ierr = PetscObjectReference((PetscObject)feti);CHKERRQ(ierr);
  if(ft->feti) {ierr = PetscObjectDereference((PetscObject)ft->feti);CHKERRQ(ierr);}
  ft->feti = feti;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPSolve_FETI"
static PetscErrorCode KSPSolve_FETI(KSP ksp)
{
  PetscErrorCode ierr;
  Mat            Amat;
  KSP_FETI       *ft = (KSP_FETI*)ksp->data;
  KSP            ksp_feti;
  
  PetscFunctionBegin;
  ierr = PCGetOperators(ksp->pc,&Amat,NULL);CHKERRQ(ierr);
  /*>>> FETI stuff */
  ierr = FETISetMat(ft->feti,Amat);CHKERRQ(ierr);
  ierr = FETISetRHS(ft->feti,ksp->vec_rhs);CHKERRQ(ierr);
  ierr = FETISolve(ft->feti,ksp->vec_sol);CHKERRQ(ierr);
  /*<<< FETI stuff */
  ierr = FETIGetKSPInterface(ft->feti,&ksp_feti);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp_feti,&ksp->reason);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp_feti,&ksp->its);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(ksp_feti,&ksp->rnorm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_FETI"
static PetscErrorCode KSPDestroy_FETI(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_FETI       *ft = (KSP_FETI*)ksp->data;

  PetscFunctionBegin;
  ierr = FETIDestroy(&ft->feti);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_FETI"
static PetscErrorCode KSPSetFromOptions_FETI(PETSC_UNUSED PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_FETI       *ft = (KSP_FETI*)ksp->data;
  
  PetscFunctionBegin;
  ierr = FETISetFromOptions(ft->feti);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
      KSPFETI - It is a wrapper for FETI methods. In this way, FETI
      methods can be used as a solver for SNES and TS solvers.

   Level: beginner

M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_FETI"
PETSC_EXTERN PetscErrorCode KSPCreate_FETI(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_FETI       *ft;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,&ft);CHKERRQ(ierr);
  ksp->data = (void*)ft;

  ft->feti       = 0;
  
  ksp->ops->setup          = KSPSetUp_FETI;
  /* ksp->ops->view           = KSPView_FETI;*/
  ksp->ops->solve          = KSPSolve_FETI;
  ksp->ops->setfromoptions = KSPSetFromOptions_FETI;
  ksp->ops->destroy        = KSPDestroy_FETI;
  
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetType(ksp->pc,PCNONE);CHKERRQ(ierr);  

  PetscFunctionReturn(0);
}

