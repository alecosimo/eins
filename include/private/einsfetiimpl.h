#if !defined(FETIIMPL_H)
#define FETIIMPL_H

#include <einsfeti.h>
#include <petscksp.h>
#include <private/einssubdomain.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool FETIRegisterAllCalled;
PETSC_EXTERN PetscErrorCode FETIRegisterAll(void);


typedef struct _FETIOps *FETIOps;
struct _FETIOps {
  PetscErrorCode (*setup)(FETI);
  PetscErrorCode (*setfromoptions)(PetscOptions*,FETI);
  PetscErrorCode (*destroy)(FETI);
  PetscErrorCode (*view)(FETI,PetscViewer);
  PetscErrorCode (*buildsolution)(FETI,Vec,Vec*);
};

/*
   FETI context
*/
struct _p_FETI {
  PETSCHEADER(struct _FETIOps);
  /* Subdomain info*/
  Subdomain        subdomain;
  Mat              A_II,A_BB,A_IB;
  /* Scaling info*/
  Mat              Wscaling;
  ScalingType      scalingType;
  /* Common attributes for the interface problem*/
  Vec              lambda_local;
  PetscInt         n_lambda;
  VecScatter       l2g_lambda;
  Mat              F;
  Vec              d;
  KSP              ksp_interface;
  KSPType          ksp_type_interface;
  PCType           pc_type_interface;
  /* B matrices*/
  Mat              B_delta,B_Ddelta;
  /* Internal use of the class*/
  PetscInt         setupcalled;
  PetscInt         setfromoptionscalled;
  void             *data;
};

PETSC_EXTERN PetscLogEvent FETI_SetUp;

#endif/* FETIIMPL_H */
