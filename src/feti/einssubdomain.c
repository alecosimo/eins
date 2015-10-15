#include <private/einssubdomain.h>
#include <petsc/private/petscimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SubdomainDestroy"
/*@
  SubdomainDestroy - Destroy the allocated structures. 

  Level: developer

.keywords: FETI
.seealso: FETICreate()
@*/
PetscErrorCode  SubdomainDestroy(Subdomain *_sd)
{
  Subdomain      sd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_sd,1);
  sd = *_sd; *_sd = NULL;
  if (!sd) PetscFunctionReturn(0);
  if (--sd->refct > 0) PetscFunctionReturn(0);
  /* Free memory*/
  ierr = PetscFree(sd);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainCheckState"
/*@
  SubdomainCheckState - Check if every structure needed by FETI has been initialized.

  Level: developer

.keywords: FETI
.seealso: FETISetUp()
@*/
PetscErrorCode  SubdomainCheckState(Subdomain sd)
{
  PetscFunctionBegin;
  if (!sd->localA)   SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Local system matrix must be first defined");
  if (!sd->localRHS) SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Local system RHS must be first defined");
  if (!sd->mapping)  SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Subdomain: Mapping from local to global DOF numbering must be first defined");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainCreate"
/*@
  SubdomainCreate - Creates the basic structures for dealing with subdomain information, such
  as the local system matrix, the local rhs and the mapping from local to global numering of
  DOFs.

  Level: developer

.keywords: FETI
.seealso: FETICreate()
@*/
PetscErrorCode  SubdomainCreate(Subdomain *_sd)
{
  Subdomain      sd;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_sd,1);
  ierr = PetscCalloc1(1,&sd);CHKERRQ(ierr);
  *_sd = sd; sd->refct = 1;

  PetscFunctionReturn(0);  
}
