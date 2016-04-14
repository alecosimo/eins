#if !defined(EINSMAT_H)
#define EINSMAT_H

#include <petscmat.h>

#define MATSHELLUNASM  "matshellunasm"

PETSC_EXTERN PetscErrorCode MatShellUnAsmGetContext(Mat,void*);
PETSC_EXTERN PetscErrorCode MatCreate_ShellUnAsm(Mat);
PETSC_EXTERN PetscErrorCode MatCreateShellUnAsm(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,void*,Mat*);
PETSC_EXTERN PetscErrorCode MatShellUnAsmSetContext(Mat,void*);
PETSC_EXTERN PetscErrorCode MatShellUnAsmSetOperation(Mat,MatOperation,void(*)(void));
PETSC_EXTERN PetscErrorCode MatShellUnAsmGetOperation(Mat,MatOperation,void(**)(void));

PETSC_EXTERN PetscErrorCode MatCreateLGMat(MPI_Comm,PetscInt,PetscInt,Mat,Mat*);
  
#endif/* EINSMAT_H*/
