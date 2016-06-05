#if !defined(EINSFETIONLY_H)
#define EINSFETIONLY_H

#include <petsc/private/snesimpl.h>
#include <einsfeti.h>

typedef struct {
  FETI           feti;
} SNES_FETIONLY;

#endif/* EINSFETIONLY_H*/
