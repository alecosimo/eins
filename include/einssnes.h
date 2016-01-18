#if !defined(EINSSNES_H)
#define EINSSNES_H

#include <petscsnes.h>
        
#define SNESFETIONLY      "fetionly"

PETSC_EXTERN PetscErrorCode SNESGetFETI(SNES,FETI*);

#endif/* EINSSNES_H*/
