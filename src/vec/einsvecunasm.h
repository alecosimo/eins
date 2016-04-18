#if !defined(EINSVECUNASM_H)
#define EINSVECUNASM_H

#include <private/einsvecimpl.h>
#include <einssys.h>
#include <einsfeti.h>

typedef struct {
  Vec      vlocal;
  Vec      multiplicity;
  PetscInt *local_sizes;
  FETI     feti;
}Vec_UNASM;


#endif/* EINSVECUNASM_H*/
