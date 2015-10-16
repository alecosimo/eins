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
  
  PetscInt n;                /* number of nodes (interior+interface) in this subdomain */
  PetscInt n_B;              /* number of interface nodes in this subdomain */
  IS       is_B_local,       /* local (sequential) index sets for interface (B) and interior (I) nodes */
           is_I_local,
           is_B_global,
           is_I_global;

  Mat localA;                /* local matrix*/
  Vec localRHS;              /* local RHS */

  VecScatter  global_to_D;        /* scattering context from global to local interior nodes */
  VecScatter  N_to_B;             /* scattering context from all local nodes to local interface nodes */
  VecScatter  global_to_B;        /* scattering context from global to local interface nodes */

  ISLocalToGlobalMapping mapping;
  PetscInt  n_neigh;     /* number of neighbours this subdomain has (by now, INCLUDING OR NOT the subdomain itself). */
                         /* Once this is definitively decided, the code can be simplifies and some if's eliminated.  */
  PetscInt *neigh;       /* list of neighbouring subdomains                                                          */
  PetscInt *n_shared;    /* n_shared[j] is the number of nodes shared with subdomain neigh[j]                        */
  PetscInt **shared;     /* shared[j][i] is the local index of the i-th node shared with subdomain neigh[j]          */
  /* It is necessary some consistency in the                                                  */
  /* numbering of the shared edges from each side.                                            */
  /* For instance:                                                                            */
  /*                                                                                          */
  /* +-------+-------+                                                                        */
  /* |   k   |   l   | subdomains k and l are neighbours                                      */
  /* +-------+-------+                                                                        */
  /*                                                                                          */
  /* Let i and j be s.t. proc[k].neigh[i]==l and                                              */
  /*                     proc[l].neigh[j]==k.                                                 */
  /*                                                                                          */
  /* We need:                                                                                 */
  /* proc[k].loc_to_glob(proc[k].shared[i][m]) == proc[l].loc_to_glob(proc[l].shared[j][m])   */
  /* for all 0 <= m < proc[k].n_shared[i], or equiv'ly, for all 0 <= m < proc[l].n_shared[j]  */
  ISLocalToGlobalMapping BtoNmap;
};

PETSC_EXTERN PetscErrorCode SubdomainDestroy(Subdomain*);
PETSC_EXTERN PetscErrorCode SubdomainCreate(Subdomain*);
PETSC_EXTERN PetscErrorCode SubdomainCheckState(Subdomain);
PETSC_EXTERN PetscErrorCode SubdomainSetLocalMat(Subdomain,Mat);
PETSC_EXTERN PetscErrorCode SubdomainSetLocalVec(Subdomain,Vec);
PETSC_EXTERN PetscErrorCode SubdomainSetMapping(Subdomain,ISLocalToGlobalMapping);

#endif/* SUBDOMAIN_H*/
