#if !defined(EINSVECIMPL_H)
#define EINSVECIMPL_H

#include <einsvec.h>
#include <petsc/private/vecimpl.h>
#include <petscmat.h>


typedef struct _VecExchangeOps *VecExchangeOps;
struct _VecExchangeOps {
  PetscErrorCode (*begin)(VecExchange,Vec,InsertMode);
  PetscErrorCode (*end)(VecExchange,Vec,InsertMode);
  PetscErrorCode (*destroy)(VecExchange);
  PetscErrorCode (*view)(VecExchange,PetscViewer);
};

struct _p_VecExchange {
  PETSCHEADER(struct _VecExchangeOps);
  /* Data about neighbors */
  PetscInt        n_neigh;
  PetscInt        *neigh, *n_shared, **shared;
  PetscCopyMode   copy_mode;

  /* MPI communications */
  PetscInt        n_reqs;
  MPI_Request     *s_reqs,*r_reqs;
  PetscScalar     **work_vecs;
  PetscScalar     **send_arrays;
};

PETSC_EXTERN PetscErrorCode VecCreate_UNASM(Vec);

#endif/* EINSVECIMPL_H*/
