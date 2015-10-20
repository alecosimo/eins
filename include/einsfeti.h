#if !defined(FETI_H)
#define FETI_H

#include <petscsys.h>
#include <petscmat.h>

/*S
     FETI - Abstract PETSc object that manages all FETI methods

   Level: beginner

  Concepts: FETI methods

.seealso:  FETICreate(), FETISetType(), FETIType (for list of available types)
S*/
typedef struct _p_FETI* FETI;


/*J
    FETIType - String with the name of a FETI method.

   Level: beginner

   FETIRegister() is used to register FETI methods that are then accessible via FETISetType()

.seealso: FETISetType(), FETI, FETICreate(), FETIRegister(), FETISetFromOptions()
J*/
typedef const char* FETIType;
#define FETINONE            "none"
#define FETI1               "feti1"


/*
    FETIList contains the list of FETI methods currently registered
   These are added with FETIRegister()
*/
PETSC_EXTERN PetscFunctionList FETIList;

/* Logging support */
PETSC_EXTERN PetscClassId FETI_CLASSID;

/* Scaling type*/ 
typedef enum {
  RHO_SCALING=0,        /* rho-scaling */
  MULTIPLICITY_SCALING, /* Multiplicity scaling */
} ScalingType;

/* QMatrix type*/ 
typedef enum {
  NONE=0,              /* none */
  DIRCHLET,            /* Dirichlet preconditioner */
  LUMPED,              /* Lumped preconditioner */
  MATCH_PCFETI         /* Equal to the preconditioner for the interface problem */
} QMatrixType;


PETSC_EXTERN PetscErrorCode FETICreate(MPI_Comm,FETI*);
PETSC_EXTERN PetscErrorCode FETIRegister(const char[],PetscErrorCode(*)(FETI));
PETSC_EXTERN PetscErrorCode FETISetType(FETI,FETIType);
PETSC_EXTERN PetscErrorCode FETIGetType(FETI,FETIType*);
PETSC_EXTERN PetscErrorCode FETISetFromOptions(FETI);
PETSC_EXTERN PetscErrorCode FETIDestroy(FETI*);
PETSC_EXTERN PetscErrorCode FETISetUp(FETI);
PETSC_EXTERN PetscErrorCode FETISetMapping(FETI,ISLocalToGlobalMapping);
PETSC_EXTERN PetscErrorCode FETISetLocalRHS(FETI,Vec);
PETSC_EXTERN PetscErrorCode FETISetLocalMat(FETI,Mat);
PETSC_EXTERN PetscErrorCode FETICreateGlobalWorkingVec(FETI,Vec);

#endif/* FETI_H*/
