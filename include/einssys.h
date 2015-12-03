#if !defined(EINSSYS_H)
#define EINSSYS_H

#include <petsc.h>
#include <einsConfig.h>
#include <petscviewerhdf5.h>

#define NAMEDOMAIN "DOM%07i"

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
#if defined(PETSC_HAVE_HDF5)
PETSC_EXTERN PetscErrorCode HDF5ArrayView(PetscInt,const void*,hid_t,PetscInt*,PetscViewer);
#endif

#endif/* EINSSYS_H*/
