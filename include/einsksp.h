#if !defined(EINSKSP_H)
#define EINSKSP_H

#include <einsfeti.h>
#include <petscksp.h>

#define KSPPJGMRES "pjgmres"
#define KSPPJCG    "pjcg"
#define KSPFETI    "feti"

PETSC_EXTERN PetscErrorCode KSPSetProjection(KSP,PetscErrorCode (*)(void*,Vec,Vec),void*);
PETSC_EXTERN PetscErrorCode KSPSetReProjection(KSP,PetscErrorCode (*)(void*,Vec,Vec),void*);
PETSC_EXTERN PetscErrorCode KSPPJGMRESMonitorKrylov(KSP,PETSC_UNUSED PetscInt,PETSC_UNUSED PetscReal,void*);
PETSC_EXTERN PetscErrorCode KSPGetResidual(KSP,Vec*);
PETSC_EXTERN PetscErrorCode KSPSetMonitorASCIIFile(KSP,const char*);
PETSC_EXTERN PetscErrorCode KSPMonitorWriteToASCIIFile(KSP,PetscInt,PetscReal,void*);
  
/* KSPFETI */
PETSC_EXTERN PetscErrorCode KSPSetFETI(KSP,FETI);
PETSC_EXTERN PetscErrorCode KSPGetFETI(KSP,FETI*);

/*E

  KSPPJCGTruncationType - Define how stored directions are used to orthogonalize in PJCG

  KSP_PJCG_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_PJCG_TRUNC_TYPE_NOTAY uses the last max(1,mod(i,mmax)) stored directions at iteration i=0,1..

   Level: intermediate

.seealso : KSPPJCG,KSPPJCGSetTruncationType(),KSPPJCGGetTruncationType()

E*/
typedef enum {KSP_PJCG_TRUNC_TYPE_STANDARD,KSP_PJCG_TRUNC_TYPE_NOTAY} KSPPJCGTruncationType;
PETSC_EXTERN const char *const KSPPJCGTruncationTypes[];

PETSC_EXTERN PetscErrorCode KSPPJCGSetMmax(KSP,PetscInt);
PETSC_EXTERN PetscErrorCode KSPPJCGGetMmax(KSP,PetscInt*);
PETSC_EXTERN PetscErrorCode KSPPJCGSetNprealloc(KSP,PetscInt);
PETSC_EXTERN PetscErrorCode KSPPJCGGetNprealloc(KSP,PetscInt*);
PETSC_EXTERN PetscErrorCode KSPPJCGSetTruncationType(KSP,KSPPJCGTruncationType);
PETSC_EXTERN PetscErrorCode KSPPJCGGetTruncationType(KSP,KSPPJCGTruncationType*);


#endif/* EINSKSP_H*/
