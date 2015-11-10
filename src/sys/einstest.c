#include <einstest.h>
#include <petsc/private/vecimpl.h>

#undef __FUNCT__
#define __FUNCT__ "TestAssertScalars"
/*@
   TestAssertScalars - Asserts if two scalars are equal up to a given tolerance.

   Input: 
.  a   -  Scalar to compare
.  b   -  Scalar to compare
.  tol -  Tolerance

   Level: developer

.keywords: Unit test
@*/
PETSC_EXTERN PetscErrorCode TestAssertScalars(PetscScalar a,PetscScalar b,PetscScalar tol)
{
  PetscFunctionBegin;
  if (PetscAbs(a-b)>tol)
    SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_SIG,"Error: scalars are not equal: a=%g, b=%g, tol=%g",a,b,tol); 
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TestAssertVectors"
/*@
   TestAssertVectors - Asserts if two vectors have point-wise equal entries up to a given tolerance.

   Input: 
.  a   -  Scalar to compare
.  b   -  Scalar to compare
.  tol -  Tolerance

   Level: developer

.keywords: Unit test
@*/
PETSC_EXTERN PetscErrorCode TestAssertVectors(Vec a,Vec b,PetscScalar tol)
{
  PetscErrorCode      ierr;
  const PetscScalar   *v1,*v2;
  PetscInt            i;
  
  PetscFunctionBegin;
  if (a->map->n!=b->map->n)
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_SIG,"Error: vectors have different local size: a->map->n=%d, b=%d",a->map->n,b->map->n); 
  ierr = VecGetArrayRead(a,&v1);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b,&v2);CHKERRQ(ierr);
  for (i=0;i<a->map->n;i++)
    if (PetscAbs(v1[i]-v2[i])>tol)
      SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_SIG,"Error: vector local entry %d is not equal: a=%g, b=%g, tol=%g",i,v1[i],v2[i],tol); 
  ierr = VecRestoreArrayRead(a,&v1);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b,&v2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
