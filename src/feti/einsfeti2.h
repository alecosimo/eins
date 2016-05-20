#if !defined(FETI2_H)
#define FETI2_H

#include <private/einsfetiimpl.h>
#include <einssys.h>

#if defined(HAVE_SLEPC)
#include <slepceps.h>

PETSC_INTERN PetscErrorCode FETI2ComputeMatrixG_GENEO(FETI);
PETSC_INTERN PetscErrorCode FETISetUp_FETI2_GENEO(FETI);
PETSC_INTERN PetscErrorCode FETICreate_FETI2_GENEO(FETI);
PETSC_INTERN PetscErrorCode FETIDestroy_FETI2_GENEO(FETI);

typedef struct {
  PC            pc_dirichlet;
  PC            pc; /* this is the preconditioner specified for using in the FETI solver. It can be the same as pc_dirichlet */
  EPS           eps;
  Mat           Bg; /* this is a shell matrix for defining the B of the generalized eigenvalue problem, mainly B^T*S*B */
  Vec           vec_lb1,vec_lb2; /* working vectors */
} GENEO_C; /* underscore "_C" becuase it is for defining a coarse space */

struct _GENEOMat_ctx {
  FETI    ft;
  GENEO_C *gn;
};
typedef struct _GENEOMat_ctx *GENEOMat_ctx;

#endif


/* Private context for the FETI-2 method.  */
typedef struct {
  Vec          res_interface;   /* residual of the interface problem, i.e., r=d-F*lambda */
  
  /* Coarse problem stuff */
  PetscMPIInt  *displ;          /* Entry i specifies the displacement at which to place the incoming data from process i in gather operations */
                                         /* displ is relative to the global communicator*/
  PetscInt     total_rbm;       /* total number of rigid body modes */
  PetscInt     max_n_cs;       /* the maximum of the number of rbm from me and my neighbours */
  Mat          localG;          /* local G matrix (current processor) */
  Mat          rbm;             /* matrix storing rigid body modes */ 
  Mat          *Gholder;        /* each entry is one neighbour's localG matrix. The order follows, the order of ft2->neighs_lb. */
                                /* The actual values are stored in the array "PetscScalar  *matrices" */
  PetscScalar  *matrices;       /* array for storing the values of the matrices in Gholder */
  PetscInt     n_Gholder;       /* number of floating neighbours */
  PetscInt     **neigh_holder;  /* Entry neigh_holder[i][0] corresponds to rank of the processor of the neighbour with matrix localG stored in Gholder[i]. 
				   Entry neigh_holder[i][1] corresponds to the index in which neigh_holder[i][0] is listed in the array ft->neigh_lb. */
  Mat          coarse_problem;
  Mat          F_coarse;        /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */
  KSP          ksp_coarse;

  /* To compute rigid body modes */
  FETI2IStiffness stiffnessFun;
  Mat             stiffness_mat;
  void            *stiffness_ctx;
  PetscBool       computeRBM;
  Mat             F_rbm; /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */
  KSP             ksp_rbm;

  /* Coarse grid types */
  CoarseGridType  coarseGType;
#if defined(HAVE_SLEPC)
  GENEO_C         *geneo;
#endif
  
  /* data for the coarse problem built in FETI2SetUpCoarseProblem_RBM */
  MPI_Request    *send2_reqs,*recv2_reqs;
  PetscInt       **neighs2,*n_neighs2; /* arrays to save which are the neighbours of neighbours */
  Mat            *FGholder;  /* each entry is one neighbour's local F*G. The order follows, the order of ft2->neigh_lb. */
  PetscInt       n_FGholder;
  PetscScalar    *bufferRHS,*bufferX,*bufferG; /* matrix data in column major order */
  PetscInt       localnnz; /* local nonzeros */
  PetscScalar    *fgmatrices; /* F*G matrices computed by my neighbors */
  PetscInt       n_send2,n_recv2; /* number of sends and receives when communicating F*G's */
  Mat            *sum_mats;   /* petsc matrices storing the sum: F_s*G, the actual data is stored in bufferPSum */
  PetscScalar    *bufferPSum; /* buffer for storing the data of F_s*G */
  PetscInt       n_sum_mats,*i2rank;
  Mat            local_rows_matrix; /* matrix storing the rows of the matrix of the coarse problem corresponding to the current processor */
  PetscInt       *c_coarse,*r_coarse,*c_count; /* used for storing rows and columns indices of the matrix of the coarse problem */
  PetscInt       *n_cs_comm; /* each entry i is the number of RBM of subdomain i */
  
} FETI_2;


#endif /* FETI2_H */
