#if !defined(EINSSNES_H)
#define EINSSNES_H

#include <petscsnes.h>
#include <einsfeti.h>

#define SNESFETIONLY      "fetionly"

PETSC_EXTERN PetscErrorCode SNESGetFETIContext(SNES,FETI*);
  
#endif/* EINSSNES_H*/
