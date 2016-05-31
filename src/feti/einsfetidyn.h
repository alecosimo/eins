#if !defined(FETIDYN_H)
#define FETIDYN_H

#include <private/einsfetiimpl.h>
#include <einssys.h>


/* Private context for the FETI method for dynamics.  */
typedef struct {
  PetscBool empty;      
} FETI_DYN;


#endif /* FETIDYN_H */
