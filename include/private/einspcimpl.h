#if !defined(EINSPCIMPL_H)
#define EINSPCIMPL_H

#include <einspc.h>
#include <einsfeti.h>
#include <petsc/private/pcimpl.h>

#define PCFETIHEADER                 \
  FETI ft;                           \
  /* MPI communications */           \
  PetscInt        n_reqs;            \
  MPI_Request     *s_reqs,*r_reqs;   \
  PetscScalar     **work_vecs;       \
  IS              *isindex;          \
  PetscBool       *pnc;              \
  Vec             vec1;              \
  Mat             RHS,Xs;	     \
  PetscScalar     *buffer_rhs,*buffer_xs;   \
  PetscScalar     **work_vecs_r;           \
  MPI_Comm        comm;

typedef struct {
  PCFETIHEADER
} PCFT_BASE;


PETSC_EXTERN PetscErrorCode PCCreate_DIRICHLET(PC);
PETSC_EXTERN PetscErrorCode PCCreate_LUMPED(PC);

PetscErrorCode PCDeAllocateFETIWorkVecs_Private(PC);
PetscErrorCode PCAllocateFETIWorkVecs_Private(PC,FETI);
PetscErrorCode PCAllocateCommunication_Private(PC,PetscInt*);

#endif/* EINSPCIMPL_H*/
