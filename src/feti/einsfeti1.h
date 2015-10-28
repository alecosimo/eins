#if !defined(FET1_H)
#define FET1_H

#include <private/einsfetiimpl.h>

/* Private context for the FETI-1 method.  */
typedef struct {
  Mat          F_neumann; /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */

  /* Coarse problem stuff */
  const char*  coarse_options;  /* for setting default options database for coarse problem */
  PetscMPIInt  *displ;          /* Entry i specifies the displacement at which to place the incoming data from process i in gather operations */
                                /* it is relative to communicator floatingComm */
  PetscMPIInt  *count_rbm;      /* Entry i specifies the number of elements to be received from process i. It is equal to the n_rbm of each process */
                                /* it is relative to communicator floatingComm */
  PetscInt     total_rbm;       /* total number of rigid body modes */
  PetscInt     n_rbm;           /* local number of rigid body modes */
  MPI_Comm     floatingComm;    /* communicator for floating substructures */
  PetscMPIInt  *neigh_floating; /* neighbours translated to the new communicator floatingComm numbering */
  Mat          localG;          /* local G matrix (current processor) */
  Mat          *Gholder;        /* each entry is one neighbour's localG matrix. The order follows, the order of ft1->neighs_lb */
  PetscScalar  *matrices;       /* array for storing the values of the matrices "stored" in Gholder */
  PetscInt     n_Gholder;       /* number of floating neighbours */
  Vec          local_e;
  Mat          coarse_problem;
  Mat          F_coarse;        /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */
  KSP          ksp_coarse;
  PetscBool    destroy_coarse;  /* destroy coarse matrix after factorization? */
  
} FETI_1;


#endif /* FET1_H */
