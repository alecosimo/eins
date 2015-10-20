#if !defined(EINSSYS_H)
#define EINSSYS_H

#include <petsc.h>

/* ---------------------------------------------------------------- */
/*   Initialization of EINS */

PETSC_EXTERN PetscErrorCode EinsInitialize(int*,char***,const char[],const char[]);
PETSC_EXTERN PetscErrorCode EinsInitialized(PetscBool *);
PETSC_EXTERN PetscErrorCode EinsFinalize(void);
PETSC_EXTERN PetscErrorCode EinsFinalized(PetscBool *);

/* ---------------------------------------------------------------- */
/*   Some utils */
PETSC_EXTERN PetscErrorCode VecSeqViewSynchronized(Vec);
PETSC_EXTERN PetscErrorCode MatSeqViewSynchronized(Mat);
PETSC_EXTERN PetscErrorCode ISSubsetNumbering(IS,IS,PetscInt*,IS*);
PETSC_EXTERN PetscErrorCode ISCreateMPIVec(MPI_Comm,PetscInt,ISLocalToGlobalMapping,Vec*);

#endif/* EINSSYS_H*/