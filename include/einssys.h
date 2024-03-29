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
PETSC_EXTERN PetscErrorCode initializeVecSeqToRank(MPI_Comm,Vec);

/* ---------------------------------------------------------------- */
/*   HDF5 utils */
#if defined(PETSC_HAVE_HDF5)
PETSC_EXTERN PetscErrorCode HDF5ArrayView(PetscInt,const void*,hid_t,PetscInt**,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5WriteGroupAttribute(PetscViewer,const char[],PetscDataType,const void*);
PETSC_EXTERN PetscErrorCode PetscViewerHDF5CreateSoftLink(PetscViewer,const char[],const char[]);
#endif


#if defined(__cplusplus)
#define EINS_INTERN extern "C" PETSC_VISIBILITY_INTERNAL
#else
#define EINS_INTERN extern PETSC_VISIBILITY_INTERNAL
#endif

#endif/* EINSSYS_H*/
