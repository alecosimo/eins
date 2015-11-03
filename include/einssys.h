#if !defined(EINSSYS_H)
#define EINSSYS_H

#include <petsc.h>
#include <einsConfig.h>

/* ---------------------------------------------------------------- */
/*   Initialization of EINS */

PETSC_EXTERN PetscErrorCode EinsInitialize(int*,char***,const char[],const char[]);
PETSC_EXTERN PetscErrorCode EinsInitialized(PetscBool *);
PETSC_EXTERN PetscErrorCode EinsFinalize(void);
PETSC_EXTERN PetscErrorCode EinsFinalized(PetscBool *);

/* ---------------------------------------------------------------- */
/*   Some utils */
PETSC_EXTERN PetscErrorCode VecSeqViewSynchronized(MPI_Comm,Vec);
PETSC_EXTERN PetscErrorCode MatSeqViewSynchronized(MPI_Comm,Mat);
PETSC_EXTERN PetscErrorCode ISSubsetNumbering(IS,IS,PetscInt*,IS*);
PETSC_EXTERN PetscErrorCode ISCreateMPIVec(MPI_Comm,PetscInt,ISLocalToGlobalMapping,Vec*);

#endif/* EINSSYS_H*/
