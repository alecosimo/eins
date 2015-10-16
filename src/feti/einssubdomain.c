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
  ierr = MatDestroy(&sd->localA);CHKERRQ(ierr);
  ierr = VecDestroy(&sd->localRHS);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&sd->mapping);CHKERRQ(ierr);
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


#undef  __FUNCT__
#define __FUNCT__ "SubdomainSetLocalRHS"
/*@
   SubdomainSetLocalRHS - Sets the local system RHS for the current process.

   Input Parameter:
.  sd  - The Subdomain context
.  rhs - The local system rhs

   Level: developer

.keywords: Subdomain, local system rhs

.seealso: SubdomainSetLocalMat(), SubdomainSetMapping()
@*/
PetscErrorCode SubdomainSetLocalRHS(Subdomain sd,Vec rhs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  /* this rutine is called from outside with a valid rhs*/
  ierr = VecDestroy(&sd->localRHS);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)rhs);CHKERRQ(ierr);
  sd->localRHS = rhs;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SubdomainSetLocalMat"
/*@
   SubdomainSetLocalMat - Sets the local system matrix for the current process.

   Input Parameter:
.  sd    - The Subdomain context
.  local_mat - The local system matrix

   Level: beginner

.keywords: Subdomain, local system matrix

.seealso: SubdomainSetLocalRHS(), SubdomainSetMapping()
@*/
PetscErrorCode SubdomainSetLocalMat(Subdomain sd,Mat local_mat)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  /* this rutine is called from outside with a valid local_mat*/
  ierr = MatDestroy(&sd->localA);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)local_mat);CHKERRQ(ierr);
  sd->localA = local_mat;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "SubdomainSetMapping"
/*@
   SubdomainSetMapping - Sets the mapping from local to global numbering of DOFs

   Input Parameter:
.  sd    - The Subdomain context
.  isg2l - A mapping from local to global numering of DOFs

   Level: developer

.keywords: Subdomain, local to global numbering of DOFs

.seealso: SubdomainSetLocalRHS(), SubdomainSetLocalMat()
@*/
PetscErrorCode SubdomainSetMapping(Subdomain sd,ISLocalToGlobalMapping isg2l)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(sd,1);
  /* this rutine is called from outside with a valid isg2l*/
  ierr = ISLocalToGlobalMappingDestroy(&sd->mapping);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)isg2l);CHKERRQ(ierr);
  sd->mapping = isg2l;
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
