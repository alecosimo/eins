#if !defined(EINSVECUNASM_H)
#define EINSVECUNASM_H

#include <private/einsvecimpl.h>

#define NAMEDOMAIN "DOM%07i"

typedef struct {
  Vec      vlocal;
  Vec      multiplicity;
  PetscInt *local_sizes;
}Vec_UNASM;


#endif/* EINSVECUNASM_H*/
