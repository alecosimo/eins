#if !defined(EINSSNES_H)
#define EINSSNES_H

#include <petscsnes.h>
#include <einsfeti.h>

#define SNESFETIONLY      "fetionly"
#define SNESNLFETI        "nlfeti" /* nonlinear-FETI */

PETSC_EXTERN PetscErrorCode SNESGetFETIContext(SNES,FETI*);
  
#endif/* EINSSNES_H*/
