#if !defined(EINS_H)
#define EINS_H

#include <petsc.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/petscimpl.h>
#else
#include <petsc/private/petscimpl.h>
#endif


/* ---------------------------------------------------------------- */
/*   Initialization of EINS */

PETSC_EXTERN PetscErrorCode EinsInitialize(int*,char***,const char[],const char[]);
PETSC_EXTERN PetscErrorCode EinsInitialized(PetscBool *);
PETSC_EXTERN PetscErrorCode EinsFinalize(void);
PETSC_EXTERN PetscErrorCode EinsFinalized(PetscBool *);

/* ---------------------------------------------------------------- */

#define KSPPJGMRES "pjgmres"


#endif/*EINS_H*/
