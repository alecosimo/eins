#if !defined(PJ1LEVEL_H)
#define PJ1LEVEL_H

#include <private/einsfetiimpl.h>
#include <einssys.h>

typedef struct {
  /* Coarse problem stuff */
  MPI_Comm     floatingComm;             /* Communicator grouping processors only with floating structures */
  PetscMPIInt  *displ,*displ_f;          /* Entry i specifies the displacement at which to place the incoming data from process i in gather operations */
                                         /* displ is relative to the global communicator and displ_f to the communicator floatingComm */
  PetscMPIInt  *count_rbm,*count_f_rbm;  /* Entry i specifies the number of elements to be received from process i. It is equal to the n_cs of each process */
                                         /* count_rbm is relative to the global communicator and count_f_rbm to the communicator floatingComm */
  PetscInt     total_rbm;       /* total number of rigid body modes */
  PetscInt     max_n_cs;       /* the maximum of the number of rbm from me and my neighbours */
  Mat          *Gholder;        /* each entry is one neighbour's localG matrix. The order follows the order of ft1->neighs_lb. */
                                /* The actual values are stored in the array "PetscScalar  *matrices" */
  PetscScalar  *matrices;       /* array for storing the values of the matrices in Gholder */
  PetscInt     n_Gholder;       /* number of floating neighbours */
  PetscInt     **neigh_holder;  /* Entry neigh_holder[i][0] corresponds to rank of the processor of the neighbour with matrix localG stored in Gholder[i]. 
				   Entry neigh_holder[i][1] corresponds to the index in which neigh_holder[i][0] is listed in the array ft->neigh_lb. */
  Mat          coarse_problem;
  Mat          F_coarse;        /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */
  KSP          ksp_coarse;
  PetscBool    destroy_coarse;  /* destroy coarse matrix after factorization? */

} PJ1LEVEL;

#endif /* PJ1LEVEL_H */
