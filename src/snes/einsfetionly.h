#include <petsc/private/snesimpl.h>
#include <einsfeti.h>

typedef struct {
  FETI           feti;
  PetscBool      refresh_jacobian;
} SNES_FETIONLY;


