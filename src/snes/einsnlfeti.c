#include "einsnlfeti.h"


#undef __FUNCT__
#define __FUNCT__ "SNESNLFETIGetFETI"
/*@
   SNESNLFETIGetFETI - Gets the FETI context. 

   Input Parameters:
.  snes   - the SNES context

   Output Parameters:
.  feti   - the FETI context

   Level: basic
@*/
PETSC_EXTERN PetscErrorCode SNESNLFETIGetFETI(SNES snes,FETI *feti);
PETSC_EXTERN PetscErrorCode SNESNLFETIGetFETI(SNES snes,FETI *feti)
{
  SNES_NLFETI    *sf = (SNES_NLFETI*)snes->data;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *feti = sf->feti;
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NLFETI"
static PetscErrorCode SNESSolve_NLFETI(SNES snes)
{
  PetscErrorCode     ierr;
  SNES_NLFETI      *sf = (SNES_NLFETI*)snes->data;
  PetscInt           lits;
  Vec                Y,X,F;
  KSP                ksp;
  
  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) {
    SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);
  }
  
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;
  snes->iter                   = 0;
  snes->norm                   = 0.0;

  X = snes->vec_sol;
  F = snes->vec_func;
  Y = snes->vec_sol_update;

  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->numbermonitors) {
    PetscReal fnorm;
    ierr = FETIComputeForceNorm(sf->feti,F,NORM_2,&fnorm);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, 0);CHKERRQ(ierr);
  }

  /* Solve J Y = F, where J is Jacobian matrix */
  ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
  ierr = FETISetMat(sf->feti,snes->jacobian);CHKERRQ(ierr);
  ierr = FETISetRHS(sf->feti,F);CHKERRQ(ierr);
  ierr = FETISolve(sf->feti,Y);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITS;
  SNESCheckKSPSolve(snes);

  ierr              = FETIGetKSPInterface(sf->feti,&ksp);CHKERRQ(ierr);
  ierr              = KSPGetIterationNumber(ksp,&lits);CHKERRQ(ierr);
  snes->linear_its += lits;
  ierr              = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
  snes->iter++;

  /* Take the computed step. */
  ierr = VecAXPY(X,-1.0,Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  
  if (snes->numbermonitors) {
    PetscReal fnorm;
    ierr = FETIComputeForceNorm(sf->feti,F,NORM_2,&fnorm);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,1,fnorm);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NLFETI"
static PetscErrorCode SNESSetUp_NLFETI(SNES snes)
{
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  
  PetscFunctionBegin;
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);  
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);

  /* configure local_snes */
  ierr = SNESSetJacobian(local_snes, );CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NLFETI"
static PetscErrorCode SNESDestroy_NLFETI(SNES snes)
{
  PetscErrorCode ierr;
  SNES_NLFETI *sf = (SNES_NLFETI*)snes->data;
  
  PetscFunctionBegin;
  if(sf->feti) {ierr = FETIDestroy(&sf->feti);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NLFETI"
/*MC
      SNESNLFETI - Nonlinear-FETI solver. Newton-like iterations are of local character.

   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), NLFETI
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NLFETI(SNES snes);
PETSC_EXTERN PetscErrorCode SNESCreate_NLFETI(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NLFETI     *sf;
  
  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NLFETI;
  snes->ops->solve          = SNESSolve_NLFETI;
  snes->ops->destroy        = SNESDestroy_NLFETI;
  snes->ops->setfromoptions = 0;
  snes->ops->view           = 0;
  snes->ops->reset          = 0;

  snes->usesksp = PETSC_FALSE;
  snes->usespc  = PETSC_FALSE;

  ierr                 = PetscNewLog(snes,&sf);CHKERRQ(ierr);
  snes->data           = (void*)sf;
  ierr                 = FETICreate(PetscObjectComm((PetscObject)snes),&sf->feti);CHKERRQ(ierr);
  ierr                 = SNESCreate(PETSC_COMM_SELF,&sf->local_snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

