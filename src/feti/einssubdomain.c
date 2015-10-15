#include<include/private/einssubdomain.h>


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
PetscErrorCode  SubdomainCreate(Subdomain *sd)
{
  
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainDestroy"
/*@
  SubdomainDestroy - Destroy the allocated structures. 

  Level: developer

.keywords: FETI
.seealso: FETICreate()
@*/
PetscErrorCode  SubdomainDestroy(Subdomain *sd)
{
  
}


#undef __FUNCT__
#define __FUNCT__ "SubdomainCheckState"
/*@C
  SubdomainCheckState - Check if every structure needed by FETI has been initialized.

  Level: developer

.keywords: FETI
.seealso: FETISetUp()
@*/
PetscErrorCode  SubdomainCheckState(Subdomain *sd)
{
  PetscFunctionBegin;
  if (!sd->localA)   SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Local system matrix must be first defined");
  if (!sd->localRHS) SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Local system RHS must be first defined");
  if (!sd->mapping)  SETERRQ(MPI_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Mapping from local to global DOF numbering must be first defined");
  PetscFunctionReturn(0);
}

