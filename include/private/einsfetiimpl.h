#if !defined(FETIIMPL_H)
#define FETIIMPL_H

#include <einsfeti.h>
#include <einsvec.h>
#include <einsmat.h>
#include <petscksp.h>
#include <einspetsccompat.h>
#include <private/einssubdomain.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool FETIRegisterAllCalled;
PETSC_EXTERN PetscErrorCode FETIRegisterAll(void);
PETSC_EXTERN PetscErrorCode FETICreateFMat(FETI,void (*)(void),void (*)(void),void (*)(void));
PETSC_EXTERN PetscErrorCode FETIBuildInterfaceKSP(FETI);


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
};

PETSC_EXTERN PetscLogEvent FETI_SetUp;

#endif/* FETIIMPL_H */
