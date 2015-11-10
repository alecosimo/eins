#if !defined(EINSVEC_H)
#define EINSVEC_H

#include <petscvec.h>

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

/* VecExchange functions */
PETSC_EXTERN PetscErrorCode VecExchangeCreate(Vec,PetscInt,PetscInt*,PetscInt*,PetscInt**,PetscCopyMode,VecExchange*);
PETSC_EXTERN PetscErrorCode VecExchangeDestroy(VecExchange*);
PETSC_EXTERN PetscErrorCode VecExchangeBegin(VecExchange,Vec,InsertMode);
PETSC_EXTERN PetscErrorCode VecExchangeEnd(VecExchange,Vec,InsertMode);
/* VECUNASM functions */
PETSC_EXTERN PetscErrorCode VecUnAsmSetMultiplicity(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecUnAsmCreateMPIVec(Vec,ISLocalToGlobalMapping,CompatibilityRule,Vec*);
PETSC_EXTERN PetscErrorCode VecUnAsmGetLocalVector(Vec,Vec*);

#endif/* EINSVEC_H*/
