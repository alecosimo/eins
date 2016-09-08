#if !defined(EINSVEC_H)
#define EINSVEC_H

#include <petscvec.h>
#include <einsfeti.h>

#define VECMPIUNASM  "mpiunasm" /* globally unassembled mpi vector. Each processor redundantly 
				   owns a portion of the global unassembled vector */

typedef enum {
  COMPAT_RULE_NONE=0,  /* None                   */
  COMPAT_RULE_AVG      /* Average value          */
} CompatibilityRule;


/*S
     VecExchange - Object used to exchange data between neighboring
     processors in globally unassebled vectors.

   Level: beginner

  Concepts: exchange

.seealso:  VecExchangeCreate(), VecExchangeBegin(), VecExchangeEnd()
S*/
typedef struct _p_VecExchange*  VecExchange;

PETSC_EXTERN PetscClassId VEC_EXCHANGE_CLASSID;

/* VecScatterUA: VecScatter for unassembled vectors */
PETSC_EXTERN PetscErrorCode VecScatterUABegin(VecScatter,Vec,Vec,InsertMode,ScatterMode);
PETSC_EXTERN PetscErrorCode VecScatterUAEnd(VecScatter,Vec,Vec,InsertMode,ScatterMode);
/* VecExchange functions */
PETSC_EXTERN PetscErrorCode VecExchangeCreate(Vec,PetscInt,PetscInt*,PetscInt*,PetscInt**,PetscCopyMode,VecExchange*);
PETSC_EXTERN PetscErrorCode VecExchangeDestroy(VecExchange*);
PETSC_EXTERN PetscErrorCode VecExchangeBegin(VecExchange,Vec,InsertMode);
PETSC_EXTERN PetscErrorCode VecExchangeEnd(VecExchange,Vec,InsertMode);
PETSC_EXTERN PetscErrorCode VecGetEntriesInArrayFromLocalVector(Vec,IS,const PetscScalar*);
/* VECUNASM functions */
PETSC_EXTERN PetscErrorCode VecUnAsmSetMultiplicity(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecUnAsmCreateMPIVec(Vec,ISLocalToGlobalMapping,CompatibilityRule,Vec*);
PETSC_EXTERN PetscErrorCode VecCreateMPIUnasmWithArray(MPI_Comm,PetscInt,PetscInt,const PetscScalar[],Vec*);
PETSC_EXTERN PetscErrorCode VecCreateMPIUnasmWithLocalVec(MPI_Comm,PetscInt,PetscInt,Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecUnAsmGetLocalVector(Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecUnAsmRestoreLocalVector(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecUnAsmGetLocalVectorRead(Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecUnAsmRestoreLocalVectorRead(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecUnAsmSum(Vec,PetscScalar*);
PETSC_EXTERN PetscErrorCode VecGetFETI(Vec,FETI*);
PETSC_EXTERN PetscErrorCode VecSetFETI(Vec,FETI);

#endif/* EINSVEC_H*/
