#if !defined(FETI1_H)
#define FETI1_H

#include <private/einsfetiimpl.h>

/* Private context for the FETI-1 method.  */
typedef struct {
  Vec    res_interface;   /* residual of the interface problem, i.e., r=d-F*lambda */
  Vec    alpha_local;
  Mat    rbm;             /* matrix storing rigid body modes */
  Vec    local_e;         /* local vector with RHS projected by the rbms */
} FETI_1;


#endif /* FETI1_H */
