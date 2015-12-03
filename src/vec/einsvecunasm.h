#if !defined(EINSVECUNASM_H)
#define EINSVECUNASM_H

#include <private/einsvecimpl.h>
#include <einssys.h>

typedef struct {
  Vec      vlocal;
  Vec      multiplicity;
  PetscInt *local_sizes;
}Vec_UNASM;


#endif/* EINSVECUNASM_H*/
