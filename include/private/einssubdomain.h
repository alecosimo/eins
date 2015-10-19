#if !defined(SUBDOMAIN_H)
#define SUBDOMAIN_H

#include <petscksp.h>

typedef struct _n_Subdomain *Subdomain;

struct _n_Subdomain {
  /* In naming the variables, we adopted the following convention: */
  /* * B - stands for interface nodes;                             */
  /* * I - stands for interior nodes;                              */
  /* * D - stands for Dirichlet (by extension, refers to interior  */
  /*       nodes) and                                              */
  /* * N - stands for Neumann (by extension, refers to all local   */
  /*       nodes, interior plus interface).                        */
  /* In some cases, I or D would apply equaly well (e.g. vec1_D).  */

  PetscInt refct;            /* reference count*/
  MPI_Comm comm;
  
  PetscInt n;                /* number of nodes (interior+interface) in this subdomain */
  PetscInt n_B;              /* number of interface nodes in this subdomain */
  IS       is_B_local,       /* local (sequential) index sets for interface (B) and interior (I) nodes */
           is_I_local,
           is_B_global,
           is_I_global;

  Mat localA;                /* local matrix*/
  Vec localRHS;              /* local RHS */

  /* working vectors */
  Vec vec1_N,
      vec1_D,
      vec1_B,
      vec1_global;
  
  VecScatter  global_to_D;        /* scattering context from global to local interior nodes */
  VecScatter  N_to_B;             /* scattering context from all local nodes to local interface nodes */
  VecScatter  global_to_B;        /* scattering context from global to local interface nodes */
  
  ISLocalToGlobalMapping mapping; /* mapping from local to global numbering of nodes */
  ISLocalToGlobalMapping BtoNmap;
  PetscInt  n_neigh;         /* number of neighbours this subdomain has (by now, INCLUDING OR NOT the subdomain itself). */
  PetscInt *neigh;           /* list of neighbouring subdomains                                                          */
  PetscInt *n_shared;        /* n_shared[j] is the number of nodes shared with subdomain neigh[j]                        */
  PetscInt **shared;         /* shared[j][i] is the local index of the i-th node shared with subdomain neigh[j]          */

  PetscInt *count;           /* number of neighbors for DOF i at the iterface. I does not count itself. dim(count)=n_B   */
  PetscInt **neighbours_set; /* neighbours_set[i][j] is the number of the j-th subdomain sharing dof i-th                */

};

PETSC_EXTERN PetscErrorCode SubdomainDestroy(Subdomain*);
PETSC_EXTERN PetscErrorCode SubdomainCreate(MPI_Comm,Subdomain*);
PETSC_EXTERN PetscErrorCode SubdomainCheckState(Subdomain);
PETSC_EXTERN PetscErrorCode SubdomainSetLocalMat(Subdomain,Mat);
PETSC_EXTERN PetscErrorCode SubdomainSetLocalRHS(Subdomain,Vec);
PETSC_EXTERN PetscErrorCode SubdomainSetMapping(Subdomain,ISLocalToGlobalMapping);
PETSC_EXTERN PetscErrorCode SubdomainSetUp(Subdomain,PetscBool);
PETSC_EXTERN PetscErrorCode SubdomainCreateGlobalWorkingVec(Subdomain,Vec);

#endif/* SUBDOMAIN_H*/
