#if !defined(RBM_H)
#define RBM_H

#include <private/einsfetiimpl.h>
#include <einssys.h>

typedef enum { RBM_STATE_INITIAL,
               RBM_STATE_COMPUTED } RBMCSState;

typedef struct {
  Mat           localG; /* matrix for storing local coarse basis */
  /* To compute rigid body modes */
  FETICSRBMIStiffness stiffnessFun;
  Mat             stiffness_mat;
  void            *stiffness_ctx;
  Mat             F_rbm; /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */
  KSP             ksp_rbm;
  Mat             rbm;             /* matrix storing rigid body modes */
  RBMCSState      state;  
} RBM_CS; /* underscore "_C" becuase it is for defining a coarse space */


#endif /* RBM_H */
