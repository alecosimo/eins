#if !defined(EINSVEC_H)
#define EINSVEC_H

#include <petscvec.h>

#define VECMPIUNASM  "mpiunasm" /* globally unassembled mpi vector. Each processor redundantly 
				   owns a portion of the global unassembled vector */

typedef enum {
  COMPAT_RULE_NONE=0,  /* None                   */
  COMPAT_RULE_AVG      /* Average value          */
} CompatibilityRule;


PETSC_EXTERN PetscErrorCode VecUnAsmSetMultiplicity(Vec,Vec);
PETSC_EXTERN PetscErrorCode VecUnAsmCreateMPIVec(Vec,ISLocalToGlobalMapping,CompatibilityRule,Vec*);

#endif/* EINSVEC_H*/
