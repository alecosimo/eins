#if !defined(FET1_H)
#define FET1_H

#include <private/einsfetiimpl.h>

/* Private context for the FETI-1 method.  */
typedef struct {
  Mat          F_neumann; /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */

  /* Coarse problem stuff */
  PetscInt     n_rbm;           /* number of rigid body modes */
  Mat          localG;
  Mat          coarse_problem;
  Mat          F_coarse;        /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */
  KSP          ksp_coarse;
  PetscBool    destroy_coarse;  /* destroy coarse matrix after factorization? */
  
} FETI_1;


#endif /* FET1_H */
