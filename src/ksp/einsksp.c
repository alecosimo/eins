#include <private/einskspimpl.h>

#undef __FUNCT__
#define __FUNCT__ "KSPSetProjection"
PETSC_EXTERN PetscErrorCode KSPSetProjection(KSP ksp,PetscErrorCode (*project)(void*,Vec,Vec),void *ctx) {
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
PETSC_EXTERN PetscErrorCode KSPSetReProjection(KSP ksp,PetscErrorCode (*reproject)(void*,Vec,Vec),void *ctx) {
  KSP_PROJECTION *pj;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscUseMethod(ksp,"KSPGetProjecion_C",(KSP,KSP_PROJECTION**),(ksp,&pj));CHKERRQ(ierr);
  pj->reproject = reproject;
  pj->ctxReProj = (void*)ctx;
  PetscFunctionReturn(0);
}
