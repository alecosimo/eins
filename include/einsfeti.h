#if !defined(FETI_H)
#define FETI_H

#include <petscsys.h>
#include <petscksp.h>

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
#define FETI2               "feti2"

/*
    FETIList contains the list of FETI methods currently registered
   These are added with FETIRegister()
*/
PETSC_EXTERN PetscFunctionList FETIList;

/* Logging support */
PETSC_EXTERN PetscClassId FETI_CLASSID;

/* Scaling type*/ 
PETSC_EXTERN PetscFunctionList FETIScalingList;
typedef const char* FETIScalingType;
#define SCRHO            "scrho"
#define SCMULTIPLICITY   "scmultiplicity"
#define SCNONE           "scnone"
#define SCUNK           "scunk" /* Unknown scaling*/

/* QMatrix type*/ 
typedef enum {
  NONE=0,              /* none */
  DIRCHLET,            /* Dirichlet preconditioner */
  LUMPED,              /* Lumped preconditioner */
  MATCH_PCFETI         /* Equal to the preconditioner for the interface problem */
} QMatrixType;


typedef enum {
  NO_COARSE_GRID,
  RIGID_BODY_MODES
} CoarseGridType;
PETSC_EXTERN const char *const CoarseGridTypes[];

PETSC_EXTERN PetscErrorCode FETICreate(MPI_Comm,FETI*);
PETSC_EXTERN PetscErrorCode FETIRegister(const char[],PetscErrorCode(*)(FETI));
PETSC_EXTERN PetscErrorCode FETISetType(FETI,FETIType);
PETSC_EXTERN PetscErrorCode FETIGetType(FETI,FETIType*);
PETSC_EXTERN PetscErrorCode FETISetFromOptions(FETI);
PETSC_EXTERN PetscErrorCode FETISetInterfaceSolver(FETI,KSPType,PCType);
PETSC_EXTERN PetscErrorCode FETIDestroy(FETI*);
PETSC_EXTERN PetscErrorCode FETISetUp(FETI);
PETSC_EXTERN PetscErrorCode FETISetMappingAndGlobalSize(FETI,ISLocalToGlobalMapping,PetscInt);
PETSC_EXTERN PetscErrorCode FETISetLocalRHS(FETI,Vec);
PETSC_EXTERN PetscErrorCode FETISetLocalMat(FETI,Mat);
PETSC_EXTERN PetscErrorCode FETIFinalizePackage(void);
PETSC_EXTERN PetscErrorCode FETIInitializePackage(void);
PETSC_EXTERN PetscErrorCode FETIGetKSPInterface(FETI,KSP*);
PETSC_EXTERN PetscErrorCode FETISolve(FETI,Vec);
PETSC_EXTERN PetscErrorCode FETISetFactorizeLocalProblem(FETI,PetscBool);
PETSC_EXTERN PetscErrorCode FETISetReSetupPCInterface(FETI,PetscBool);
/* FETI1 stuff */
PETSC_EXTERN PetscErrorCode FETI1SetDefaultOptions(int*,char***,const char[]);
/* FETI2 stuff */
typedef PetscErrorCode      (*FETI2IStiffness)(FETI,Mat,void*);
PETSC_EXTERN PetscErrorCode FETI2SetDefaultOptions(int*,char***,const char[]);
PETSC_EXTERN PetscErrorCode FETI2SetStiffness(FETI,Mat,FETI2IStiffness,void*);
PETSC_EXTERN PetscErrorCode FETI2SetComputeRBM(FETI,PetscBool);
PETSC_EXTERN PetscErrorCode FETI2SetCoarseGridType(FETI,CoarseGridType);
/* scaling */
PETSC_EXTERN PetscErrorCode FETIScalingSetUp(FETI);
PETSC_EXTERN PetscErrorCode FETIScalingSetScalingFactor(FETI,PetscScalar);
PETSC_EXTERN PetscErrorCode FETIScalingSetType(FETI,FETIScalingType);

#endif/* FETI_H*/
