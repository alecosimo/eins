#if !defined(KSPIMPL_H)
#define KSPIMPL_H

#include <einsksp.h>
#include <petsc/private/kspimpl.h>

PETSC_EXTERN PetscErrorCode KSPCreate_PJGMRES(KSP);
PETSC_EXTERN PetscErrorCode KSPCreate_PJCG(KSP);

typedef struct {
  void            *ctxProj;                      /* context for projection */                            
  void            *ctxReProj;                    /* context for re-projection */                         
  PetscErrorCode (*project)(void*,Vec,Vec);      /* pointer function for performing the projection step */
  PetscErrorCode (*reproject)(void*,Vec,Vec);    /* pointer function for performing the re-projection step */
} KSP_PROJECTION;


#endif/* KSPIMPL_H*/
