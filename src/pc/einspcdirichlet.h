#if !defined(PCDIRICHLET_H)
#define PCDIRICHLET_H

#include <private/einspcimpl.h>

typedef struct {
  PCFETIHEADER
  Mat Sj;
  KSP ksp_D;
} PCFT_DIRICHLET;


#endif /* PCDIRICHLET_H */
