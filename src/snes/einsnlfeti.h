#if !defined(EINSNLFETI_H)
#define EINSNLFETI_H

#include <petsc/private/snesimpl.h>
#include <einsfeti.h>

typedef struct {
  FETI           feti;
  SNES           local_snes;
} SNES_NLFETI;

#endif/* EINSNLFETI_H*/
