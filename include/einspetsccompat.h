#if !defined(EINSPETSCCOMPAT_H)
#define EINSPETSCCOMPAT_H

#include <petsc.h>

#if PETSC_VERSION_LT(3,7,0)
#define PetscOptionItems                         PetscOptions
#define PetscObjectProcessOptionsHandlers(op,ob) PetscObjectProcessOptionsHandlers(ob)
#define PetscOptionsInsertString(op,s)           PetscOptionsInsertString(s)
#define PetscOptionsInsert(op,arc,ars,fl)        PetscOptionsInsert(arc,ars,fl)
#define PetscOptionsHasName(op,pr,nm,set)        PetscOptionsHasName(pr,nm,set)
#define PetscOptionsSetValue(op,nm,vl)           PetscOptionsSetValue(nm,vl)
#define PetscOptionsPrefixPush(op,pr)            PetscOptionsPrefixPush(pr)
#define PetscOptionsPrefixPop(op)                PetscOptionsPrefixPop()
#define PetscOptionsClearValue(op,nm)            PetscOptionsClearValue(nm)
#define PetscOptionsGetBool(op,pr,nm,vl,set)     PetscOptionsGetBool(pr,nm,vl,set)
#define PetscOptionsGetEnum(op,pr,nm,el,dv,set)  PetscOptionsGetEnum(pr,nm,el,dv,set)
#define PetscOptionsGetInt(op,pr,nm,vl,set)      PetscOptionsGetInt(pr,nm,vl,set)
#define PetscOptionsGetReal(op,pr,nm,vl,set)     PetscOptionsGetReal(pr,nm,vl,set)
#define PetscOptionsGetScalar(op,pr,nm,vl,set)   PetscOptionsGetScalar(pr,nm,vl,set)
#define PetscOptionsGetString(op,pr,nm,s,n,set)  PetscOptionsGetString(pr,nm,s,n,set)
#endif

#endif/* EINSPETSCCOMPAT_H*/
