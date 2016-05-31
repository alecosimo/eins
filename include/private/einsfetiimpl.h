#if !defined(FETIIMPL_H)
#define FETIIMPL_H

#include <einssys.h>
#include <einsfeti.h>
#include <einsvec.h>
#include <einsmat.h>
#include <petscksp.h>
#include <einspetsccompat.h>
#include <private/einssubdomain.h>
#include <petsc/private/petscimpl.h>


/* FETIPJ stuff */
typedef enum { FETIPJ_STATE_INITIAL,
               FETIPJ_STATE_NEIGH_GATHERED,
	       FETIPJ_STATE_ASSEMBLED,
               FETIPJ_STATE_FACTORIZED } FETIPJStateType;

typedef struct _FETIPJOps *FETIPJOps;
struct _FETIPJOps {
  PetscErrorCode (*destroy)(FETIPJ);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,FETIPJ);
  PetscErrorCode (*setup)(FETIPJ);
  PetscErrorCode (*gatherneighbors)(FETIPJ);
  PetscErrorCode (*assemble)(FETIPJ);
  PetscErrorCode (*factorize)(FETIPJ);
  PetscErrorCode (*initialcondition)(FETIPJ);
};

struct _p_FETIPJ {
  PETSCHEADER(struct _FETIPJOps);
  PetscInt        setupcalled;
  FETI            feti;
  FETIPJStateType state;
  void            *data;
};


/* FETICS stuff */
typedef struct _FETICSOps *FETICSOps;
struct _FETICSOps {
  PetscErrorCode (*destroy)(FETICS);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,FETICS);
  PetscErrorCode (*setup)(FETICS);
  PetscErrorCode (*computecoarsebasis)(FETICS,Mat*,Mat*);
};

struct _p_FETICS {
  PETSCHEADER(struct _FETICSOps);
  PetscInt setupcalled;             /* true if setup has been called */
  FETI     feti;
  void *data;
};


/* FETI stuff */
PETSC_EXTERN PetscBool FETIRegisterAllCalled;
PETSC_EXTERN PetscErrorCode FETIRegisterAll(void);
PETSC_EXTERN PetscErrorCode FETICreateFMat(FETI,void (*)(void),void (*)(void),void (*)(void));
PETSC_EXTERN PetscErrorCode FETIBuildInterfaceKSP(FETI);
PETSC_EXTERN PetscErrorCode FETIBuildLambdaAndB(FETI);
EINS_INTERN  PetscErrorCode MatMultFlambda_FETI(FETI,Vec,Vec);

typedef enum { FETI_STATE_INITIAL,
               FETI_STATE_SETUP_INI,
	       FETI_STATE_SETUP_END,
               FETI_STATE_SOLVED } FETIStateType;


typedef struct _FETIOps *FETIOps;
struct _FETIOps {
  PetscErrorCode (*setup)(FETI);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,FETI);
  PetscErrorCode (*destroy)(FETI);
  PetscErrorCode (*view)(FETI,PetscViewer);
  PetscErrorCode (*computesolution)(FETI,Vec);
};

struct _FETIMat_ctx {
  FETI  ft;
};
typedef struct _FETIMat_ctx *FETIMat_ctx;

/*
   FETI context
*/
struct _p_FETI {
  PETSCHEADER(struct _FETIOps);
  /* Subdomain info*/
  Subdomain        subdomain;
  /* Scaling info*/
  Vec              Wscaling; /* by now only supporting diag scaling*/
  PetscReal        scaling_factor;
  FETIScalingType  scaling_type;
  /* Common attributes for the interface problem*/
  Vec                    lambda_local;
  Vec                    lambda_global; /* global distributed (mpi) solution vector for the interface problem */
  PetscInt               n_lambda,n_lambda_local;
  Mat                    F;
  Vec                    d;
  KSP                    ksp_interface;
  KSPType                ksp_type_interface;
  PCType                 pc_type_interface;
  Mat                    F_neumann;
  /* mapping information for interface problem */
  VecExchange            exchange_lambda;
  Vec                    multiplicity; /* multiplicity of lambdas */
  ISLocalToGlobalMapping mapping_lambda;
  PetscInt               n_neigh_lb;
  PetscInt               *neigh_lb;  
  PetscInt               *n_shared_lb;
  PetscInt               **shared_lb; 
  /* B matrices*/
  Mat              B_delta,B_Ddelta;
  /* Neumann problem */
  KSP              ksp_neumann;
  /* Internal use of the class*/
  FETIStateType    state;
  PetscInt         setfromoptionscalled;
  void             *data;
  PetscBool        resetup_pc_interface;
  PetscObjectState mat_state;
  PetscInt         n_cs;           /* local number of vector for the Coarse Space */
  Mat              localG;         /* local G matrix (current processor) holding the coarse space basis at interface dofs */
  FETICS           ftcs;
  FETICSType       ftcs_type;
  FETIPJ           ftpj;
  FETIPJType       ftpj_type;
};

PETSC_EXTERN PetscLogEvent FETI_SetUp;

#endif/* FETIIMPL_H */
