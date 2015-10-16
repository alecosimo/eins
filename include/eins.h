#if !defined(EINS_H)
#define EINS_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>


/* ---------------------------------------------------------------- */
/*   Initialization of EINS */

PETSC_EXTERN PetscErrorCode EinsInitialize(int*,char***,const char[],const char[]);
PETSC_EXTERN PetscErrorCode EinsInitialized(PetscBool *);
PETSC_EXTERN PetscErrorCode EinsFinalize(void);
PETSC_EXTERN PetscErrorCode EinsFinalized(PetscBool *);

/* ---------------------------------------------------------------- */
/*   Some utils */
PETSC_EXTERN PetscErrorCode VecSeqViewSynchronized(Vec);


#define KSPPJGMRES "pjgmres"


#endif/* EINS_H*/
