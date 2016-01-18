#include "einsfetionly.h"


#undef __FUNCT__
#define __FUNCT__ "SNESGetFETI"
/*@
   SNESGetFETI - Gets the FETI context. It increments the object count, so it must 
   be destroyed by the user.

   Input Parameters:
.  snes   - the SNES context

   Output Parameters:
.  feti   - the FETI context

   Level: basic

.seealso: SNESGetFETI()
@*/
PetscErrorCode SNESGetFETI(SNES snes,FETI *feti);
PetscErrorCode SNESGetFETI(SNES snes,FETI *feti)
{
  SNES_FETIONLY    *sf = (SNES_FETIONLY*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *feti = sf->feti;
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "SNESSolve_FETIONLY"
static PetscErrorCode SNESSolve_FETIONLY(SNES snes)
{
  PetscErrorCode     ierr;
  SNES_FETIONLY      *sf = (SNES_FETIONLY*)snes->data;
  PetscInt           lits;
  Vec                Y,X,F;

  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);
  
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
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, 0);CHKERRQ(ierr);
  }

  /* Solve J Y = F, where J is Jacobian matrix */
  ierr = SNESComputeJacobian(snes,X,snes->jacobian,NULL);CHKERRQ(ierr);
  ierr = FETISetLocalMat(sf->feti,snes->jacobian);CHKERRQ(ierr);
  ierr = FETISetLocalRHS(sf->feti,F);CHKERRQ(ierr);
  ierr = FETISetUp(sf->feti);CHKERRQ(ierr);
  ierr = FETISolve(sf->feti,Y);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITS;
  SNESCheckKSPSolve(snes);

  ierr              = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
  snes->linear_its += lits;
  ierr              = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
  snes->iter++;

  /* Take the computed step. */
  ierr = VecAXPY(X,-1.0,Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->numbermonitors) {
    PetscReal fnorm;
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,1,fnorm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_FETIONLY"
static PetscErrorCode SNESSetUp_FETIONLY(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_FETIONLY"
static PetscErrorCode SNESDestroy_FETIONLY(SNES snes)
{
  PetscErrorCode ierr;
  SNES_FETIONLY *sf = (SNES_FETIONLY*)snes->data;
  
  PetscFunctionBegin;
  if(sf->feti) {ierr = FETIDestroy(&sf->feti);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESCreate_FETIONLY"
/*MC
      SNESFETIONLY - Nonlinear solver that only performs one Newton
      step using FETI solvers and does not compute any norms.  The
      main purpose of this solver is to solve linear problems using
      the SNES interface, without any additional overhead in the form
      of vector operations.

   Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESNEWTONTR, FETIONLY
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_FETIONLY(SNES snes);
PETSC_EXTERN PetscErrorCode SNESCreate_FETIONLY(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_FETIONLY     *sf;
  
  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_FETIONLY;
  snes->ops->solve          = SNESSolve_FETIONLY;
  snes->ops->destroy        = SNESDestroy_FETIONLY;
  snes->ops->setfromoptions = 0;
  snes->ops->view           = 0;
  snes->ops->reset          = 0;

  snes->usesksp = PETSC_FALSE;
  snes->usespc  = PETSC_FALSE;

  ierr          = PetscNewLog(snes,&sf);CHKERRQ(ierr);
  snes->data    = (void*)sf;
  ierr          = FETICreate(PetscObjectComm((PetscObject)snes),&sf->feti);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
