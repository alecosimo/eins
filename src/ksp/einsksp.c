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


#undef __FUNCT__
#define __FUNCT__ "KSPMonitorWriteToASCIIFile"
/*@C
   KSPMonitorWriteToASCIIFile - Print the residual norm at each iteration to an ASCII file.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).
-  dummy - a PetscViewer

   Level: intermediate
@*/
PetscErrorCode  KSPMonitorWriteToASCIIFile(PETSC_UNUSED KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer)dummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = PetscViewerASCIIPrintf(viewer,"%3D %14.12e\n",n,(double)rnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPSetMonitorASCIIFile"
/*@C
   KSPSetMonitorASCIIFile - Sets monitor for printing the residual norm at each iteration to an ASCII file.

   Collective on KSP

   Input Parameters:
+  ksp   - iterative context
-  name  - name of the file to write

   Level: intermediate
@*/
PetscErrorCode  KSPSetMonitorASCIIFile(KSP ksp,const char name[])
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
    
  PetscFunctionBegin;
  ierr   = PetscViewerCreate(PetscObjectComm((PetscObject)ksp), &viewer);CHKERRQ(ierr);
  ierr   = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr   = PetscViewerFileSetName(viewer, name);CHKERRQ(ierr);
  ierr   = KSPMonitorSet(ksp,KSPMonitorWriteToASCIIFile,viewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);
  PetscFunctionReturn(0);
}
