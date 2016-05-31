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
#define SCUNK            "scunk" /* Unknown scaling*/


/*S
   FETICS - Petsc object for handling FETI Coarse Spaces

   Level: advanced

.seealso:  FETICSCreate(), FETICSSetType()
S*/
typedef struct _p_FETICS* FETICS;

/*J
    FETICSType - String with the name of a Coarse Space

   Level: intermediate

.seealso: FETICSSetType(), FETICS
J*/
typedef const char* FETICSType;
#define CS_NONE                    "csnone"
#define CS_RIGID_BODY_MODES        "csrbm"
#define CS_GENEO_MODES             "csgeneo"

PETSC_EXTERN PetscFunctionList FETICSList;
PETSC_EXTERN PetscClassId      FETICS_CLASSID;
PETSC_EXTERN PetscBool         FETICSRegisterAllCalled;


/*S
   FETIPJ - Petsc object for handling FETI projections

   Level: advanced

.seealso:  FETIPJCreate(), FETIPJSetType()
S*/
typedef struct _p_FETIPJ* FETIPJ;

/*J
    FETIPJType - String with the name of a FETI projection

   Level: intermediate

.seealso: FETIPJSetType(), FETIPJ
J*/
typedef const char* FETIPJType;
#define PJ_NONE               "pjnone"
#define PJ_FIRST_LEVEL        "pj1level"
#define PJ_FIRST_LEVEL_Q      "pj1levelq"
#define PJ_SECOND_LEVEL       "pj2level"

PETSC_EXTERN PetscFunctionList FETIPJList;
PETSC_EXTERN PetscClassId      FETIPJ_CLASSID;
PETSC_EXTERN PetscBool         FETIPJRegisterAllCalled;

/* FETIPJ stuff */
PETSC_EXTERN PetscErrorCode FETIPJCreate(MPI_Comm,FETI,FETIPJ*);
PETSC_EXTERN PetscErrorCode FETIPJDestroy(FETIPJ*);
PETSC_EXTERN PetscErrorCode FETIPJSetType(FETIPJ,const FETIPJType);
PETSC_EXTERN PetscErrorCode FETIPJSetFromOptions(FETIPJ);
PETSC_EXTERN PetscErrorCode FETIPJGatherNeighborsCoarseBasis(FETIPJ);
PETSC_EXTERN PetscErrorCode FETIPJAssembleCoarseProblem(FETIPJ);
PETSC_EXTERN PetscErrorCode FETIPJFactorizeCoarseProblem(FETIPJ);
PETSC_EXTERN PetscErrorCode FETIPJRegister(const char[],PetscErrorCode (*)(FETIPJ));
PETSC_EXTERN PetscErrorCode FETIPJRegisterAll(void);
PETSC_EXTERN PetscErrorCode FETIPJSetUp(FETIPJ);
PETSC_EXTERN PetscErrorCode FETIPJSetFETI(FETIPJ,FETI);
PETSC_EXTERN PetscErrorCode FETIPJGetType(FETIPJ,FETIPJType*);

/* FETICS stuff */
PETSC_EXTERN PetscErrorCode FETICSCreate(MPI_Comm,FETI,FETICS*);
PETSC_EXTERN PetscErrorCode FETICSDestroy(FETICS*);
PETSC_EXTERN PetscErrorCode FETICSSetType(FETICS,const FETICSType);
PETSC_EXTERN PetscErrorCode FETICSSetFromOptions(FETICS);
PETSC_EXTERN PetscErrorCode FETICSRegister(const char[],PetscErrorCode (*)(FETICS));
PETSC_EXTERN PetscErrorCode FETICSRegisterAll(void);
PETSC_EXTERN PetscErrorCode FETICSSetUp(FETICS);
PETSC_EXTERN PetscErrorCode FETICSComputeCoarseBasis(FETICS,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode FETICSSetFETI(FETICS,FETI);
PETSC_EXTERN PetscErrorCode FETICSGetType(FETICS,FETICSType*);

/* FETICSRBM */ 
typedef PetscErrorCode      (*FETICSRBMIStiffness)(FETICS,Mat,void*);
PETSC_EXTERN PetscErrorCode FETICSRBMSetStiffnessMatrixFunction(FETI,Mat,FETICSRBMIStiffness,void*);

/* FETI stuff */
PETSC_EXTERN PetscErrorCode FETICreate(MPI_Comm,FETI*);
PETSC_EXTERN PetscErrorCode FETIRegister(const char[],PetscErrorCode(*)(FETI));
PETSC_EXTERN PetscErrorCode FETISetType(FETI,FETIType);
PETSC_EXTERN PetscErrorCode FETISetCoarseSpaceType(FETI,FETICSType);
PETSC_EXTERN PetscErrorCode FETIGetCoarseSpaceType(FETI,FETICSType*);
PETSC_EXTERN PetscErrorCode FETISetProjectionType(FETI,FETIPJType);
PETSC_EXTERN PetscErrorCode FETIGetProjectionType(FETI,FETIPJType*);
PETSC_EXTERN PetscErrorCode FETIGetType(FETI,FETIType*);
PETSC_EXTERN PetscErrorCode FETISetFromOptions(FETI);
PETSC_EXTERN PetscErrorCode FETISetInterfaceSolver(FETI,KSPType,PCType);
PETSC_EXTERN PetscErrorCode FETIDestroy(FETI*);
PETSC_EXTERN PetscErrorCode FETISetUp(FETI);
PETSC_EXTERN PetscErrorCode FETISetMappingAndGlobalSize(FETI,ISLocalToGlobalMapping,PetscInt);
PETSC_EXTERN PetscErrorCode FETISetLocalRHS(FETI,Vec);
PETSC_EXTERN PetscErrorCode FETISetLocalMat(FETI,Mat);
PETSC_EXTERN PetscErrorCode FETISetRHS(FETI,Vec);
PETSC_EXTERN PetscErrorCode FETISetMat(FETI,Mat);
PETSC_EXTERN PetscErrorCode FETIFinalizePackage(void);
PETSC_EXTERN PetscErrorCode FETIInitializePackage(void);
PETSC_EXTERN PetscErrorCode FETIGetKSPInterface(FETI,KSP*);
PETSC_EXTERN PetscErrorCode FETIGetCoarseSpace(FETI,FETICS*);
PETSC_EXTERN PetscErrorCode FETIGetProjection(FETI,FETIPJ*);
PETSC_EXTERN PetscErrorCode FETISolve(FETI,Vec);
PETSC_EXTERN PetscErrorCode FETISetFactorizeLocalProblem(FETI,PetscBool);
PETSC_EXTERN PetscErrorCode FETISetReSetupPCInterface(FETI,PetscBool);
PETSC_EXTERN PetscErrorCode FETIComputeForceNorm(FETI,Vec,NormType,PetscReal*);
PETSC_EXTERN PetscErrorCode FETIComputeForceNormLocal(FETI,Vec,NormType,PetscReal*);
/* FETI1 stuff */
PETSC_EXTERN PetscErrorCode FETI1SetDefaultOptions(int*,char***,const char[]);
/* FETI2 stuff */
PETSC_EXTERN PetscErrorCode FETI2SetDefaultOptions(int*,char***,const char[]);
/* scaling */
PETSC_EXTERN PetscErrorCode FETIScalingSetUp(FETI);
PETSC_EXTERN PetscErrorCode FETIScalingSetScalingFactor(FETI,PetscScalar);
PETSC_EXTERN PetscErrorCode FETIScalingSetType(FETI,FETIScalingType);

#endif/* FETI_H*/
