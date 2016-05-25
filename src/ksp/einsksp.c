#include <private/einskspimpl.h>


#undef __FUNCT__
#define __FUNCT__ "KSPGetResidual"
/*@
   KSPGetResidual - Gets the residual vector from the given ksp. It
   increments the reference count of the residual vector.

   Input Parameters:
.  ksp - the KSP context

   Outpur Parameters:
.  res - the residual vector

   Level: intermediate

@*/
PETSC_EXTERN PetscErrorCode KSPGetResidual(KSP ksp,Vec *res)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPGetResidual_C",(KSP,Vec*),(ksp,res));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPSetProjection"
/*@
   KSPSetProjection - Sets the projection function to use in the given ksp.

   Input Parameters:
.  ksp     - the KSP context
.  project - pointer to the function for performing the projection
.  ctx     - context for the projection function

   Level: intermediate

@*/
PETSC_EXTERN PetscErrorCode KSPSetProjection(KSP ksp,PetscErrorCode (*project)(void*,Vec,Vec),void *ctx)
{
  KSP_PROJECTION *pj;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPGetProjecion_C",(KSP,KSP_PROJECTION**),(ksp,&pj));CHKERRQ(ierr);
  pj->project = project;
  pj->ctxProj = (void*)ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPSetReProjection"
/*@
   KSPSetReProjection - Sets the function for performing the reprojection phase in the given ksp.

   Input Parameters:
.  ksp       - the KSP context
.  reproject - pointer to the function for performing the reprojection
.  ctx       - context for the reprojection function

   Level: intermediate

@*/
PETSC_EXTERN PetscErrorCode KSPSetReProjection(KSP ksp,PetscErrorCode (*reproject)(void*,Vec,Vec),void *ctx)
{
  KSP_PROJECTION *pj;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPGetProjecion_C",(KSP,KSP_PROJECTION**),(ksp,&pj));CHKERRQ(ierr);
  pj->reproject = reproject;
  pj->ctxReProj = (void*)ctx;
  PetscFunctionReturn(0);
}
