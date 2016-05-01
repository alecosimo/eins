#if !defined(EINSTSALPHA2_H)
#define EINSTSALPHA2_H
#include <petscts.h>

typedef struct {
  PetscReal stage_time;
  PetscReal shift_V;
  PetscReal shift_A;
  PetscReal scale_F;
  Vec       X0,Xa,X1;
  Vec       V0,Va,V1;
  Vec       A0,Aa,A1;

  Vec       vec_dot;

  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Gamma;
  PetscReal Beta;
  PetscInt  order;

  PetscBool adapt;
  Vec       vec_sol_prev;
  Vec       vec_dot_prev;
  Vec       vec_lte_work[2];

  TSStepStatus status;
} TS_Alpha;


#endif /* EINSTSALPHA2_H*/
