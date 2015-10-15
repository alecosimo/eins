#include <einsfeti1.h>


#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETI1"
/*
   FETIDestroy_FETI1 - Destroys the FETI-1 context

   Input Parameters:
+  ft - the FETI context

.seealso FETICreate_FETI1
 */
PetscErrorCode FETIDestroy_FETI1(FETI ft)
{
  FETI_1*        feti1 = (FETI_1*)ft->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETI1"
/*
   FETISetUp_FETI1 - Prepares the structures needed by the FETI-1 solver.

   Input Parameters:
+  ft - the FETI context

*/
PetscErrorCode FETISetUp_FETI1(FETI ft)
{
  PetscErrorCode ierr;
  FETI_1*        feti1 = (FETI_1*)ft->data;

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICreate_FETI1"
/*@
   FETI1 - Implementation of the FETI-1 method. Some comments about options can be put here!

   Level: normal

.keywords: FETI, FETI-1
@*/
PETSC_EXTERN PetscErrorCode FETICreate_FETI1(FETI ft)
{
  PetscErrorCode      ierr;
  FETI_1*             feti1;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ft,&feti1);CHKERRQ(ierr);
  ft->data  = (void*)feti1;

  /* function pointers */
  ft->ops->setup               = FETISetUp_FETI1;
  ft->ops->destroy             = FETIDestroy_FETI1;
  ft->ops->setfromoptions      = 0;//FETISetFromOptions_FETI1;
  ft->ops->view                = 0;

  PetscFunctionReturn(0);
}
