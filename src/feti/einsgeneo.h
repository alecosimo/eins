#if !defined(GENEO_H)
#define GENEO_H

#include <private/einsfetiimpl.h>
#include <einssys.h>

#if defined(HAVE_SLEPC)
#include <slepceps.h>

typedef struct {
  PC            pc; /* this is the preconditioner specified for using in the FETI solver. It can be the same as pc_dirichlet */
  PC            pc_dirichlet;
  PetscBool     flg; /* if TRUE, it indicates that pc_dirichlet==pc*/
  Mat           localG; /* matrix for storing local coarse basis */
  Mat           Ag; /* this is the matrix for the eigenvalue problem. It is a shell matrix */
  EPS           eps;
  Vec           vec_lb1,vec_lb2; /* working vectors */
} GENEO_CS; /* underscore "_CS" becuase it is for defining a coarse space */

struct _GENEOMat_ctx {
  FETI     ft;
  GENEO_CS *gn;
};
typedef struct _GENEOMat_ctx *GENEOMat_ctx;

#endif

#endif /* GENEO_H */
