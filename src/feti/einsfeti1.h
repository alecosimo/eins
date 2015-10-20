#if !defined(FET1_H)
#define FET1_H

#include <private/einsfetiimpl.h>

/* Private context for the FETI-1 method.  */
typedef struct {
  Mat       F_neumann; /* matrix object specifically suited for symbolic factorization: it must not be destroyed with MatDestroy() */
  Mat       localG;
} FETI_1;


#endif /* FET1_H */
