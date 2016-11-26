#include <private/einsfetiimpl.h>
#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "FETIScalingSetUp_multiplicity"
/*@
   FETIScalingSetUp_multiplicity - Set ups structures for multiplicity scaling

   Input Parameter:
.  ft - the FETI context

   Notes:
   It must be called after calling SubdomainSetUp()

   Level: developer

.keywords: FETI

.seealso: FETIScalingSetUp(), FETICreate(), FETISetUp(), SubdomainSetUp()
@*/
PetscErrorCode FETIScalingSetUp_multiplicity(FETI);
PetscErrorCode FETIScalingSetUp_multiplicity(FETI ft)
{
  PetscErrorCode   ierr;
  Subdomain        sd = ft->subdomain;
  PetscInt         i;
  PetscScalar      *array;
  PetscMPIInt       rank;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr = VecDuplicate(sd->vec1_B,&ft->Wscaling);CHKERRQ(ierr);
  ierr = VecGetArray(ft->Wscaling,&array);CHKERRQ(ierr);
  for ( i=0;i<sd->n_B;i++ ) { array[i]=ft->scaling_factor/(sd->count[i]+1);}
  ierr = VecRestoreArray(ft->Wscaling,&array);CHKERRQ(ierr);

  if(rank==0) {
    PetscPrintf(PETSC_COMM_SELF,"wscaling result:\n");
    VecView(ft->Wscaling, PETSC_VIEWER_STDOUT_SELF);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIScalingSetUp_none"
/*@
   FETIScalingSetUp_none - Set ups no scaling

   Input Parameter:
.  ft - the FETI context

   Notes:
   It must be called after calling SubdomainSetUp()

   Level: developer

.keywords: FETI

.seealso: FETIScalingSetUp(), FETICreate(), FETISetUp(), SubdomainSetUp()
@*/
PetscErrorCode FETIScalingSetUp_none(FETI);
PetscErrorCode FETIScalingSetUp_none(FETI ft)
{
  PetscErrorCode   ierr;
  Subdomain        sd = ft->subdomain;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr = VecDuplicate(sd->vec1_B,&ft->Wscaling);CHKERRQ(ierr);
  ierr = VecSet(ft->Wscaling,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIScalingSetType"
/*@
   FETIScalingSetType - Sets the scaling type.

   Input Parameter:
.  ft           - The FETI context
.  scaling_type - The scaling type.

   Level: beginner

.keywords: FETIScaling, FETI

.seealso: FETIScalingSetUp()
@*/
PetscErrorCode FETIScalingSetType(FETI ft, FETIScalingType scaling_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ft->scaling_type = scaling_type;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIScalingSetScalingFactor"
/*@
   FETIScalingSetScalingFactor - Sets a scaling factor different from 1.0 for the multiplicity scaling. For example, 
   for the multiplicity scaling we would have W = diag(scaling_factor/multiplicity).

   Input Parameter:
.  ft             - The FETI context
.  scaling_factor - The scaling factor.

   Level: beginner

.keywords: FETIScaling, FETI

.seealso: FETIScalingSetUp()
@*/
PetscErrorCode FETIScalingSetScalingFactor(FETI ft, PetscScalar scaling_factor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ft->scaling_factor = scaling_factor;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIScalingSetUp_rho"
/*@
   FETIScalingSetUp_rho - Creates structures for rho scaling

   Input Parameter:
.  ft - the FETI context

   Notes:
   It must be called after calling SubdomainSetUp()

   Level: developer

.keywords: FETI

.seealso: FETIScalingSetUp(), FETICreate(), FETISetUp(), SubdomainSetUp()
@*/
PetscErrorCode FETIScalingSetUp_rho(FETI);
PetscErrorCode FETIScalingSetUp_rho(FETI ft)
{
  PetscErrorCode   ierr;
  PetscMPIInt       rank;
  Subdomain        sd = ft->subdomain;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr = VecDuplicate(sd->vec1_B,&ft->Wscaling);CHKERRQ(ierr);
  ierr = MatGetDiagonal(sd->localA,sd->vec1_N);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_N,ft->Wscaling,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_N,ft->Wscaling,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  
  ierr = VecSet(sd->vec1_global,0.0);CHKERRQ(ierr);

  /* if(rank==1) { */
  /*   PetscPrintf(PETSC_COMM_SELF,"wscaling result: rank %d\n",rank); */
  /*   VecView(ft->Wscaling, PETSC_VIEWER_STDOUT_SELF); */
  /* } */

  ierr = VecScatterUABegin(sd->N_to_B,ft->Wscaling,sd->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterUAEnd(sd->N_to_B,ft->Wscaling,sd->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecExchangeBegin(sd->exchange_vec1global,sd->vec1_global,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(sd->exchange_vec1global,sd->vec1_global,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecScatterUABegin(sd->N_to_B,sd->vec1_global,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterUAEnd(sd->N_to_B,sd->vec1_global,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(ft->Wscaling,ft->Wscaling,sd->vec1_B);CHKERRQ(ierr);
  /* if(rank==0) { */
  /*   PetscPrintf(PETSC_COMM_SELF,"wscaling result: rank %d\n",rank); */
  /*   VecView(ft->Wscaling, PETSC_VIEWER_STDOUT_SELF); */
  /* } */

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIScalingSetUp"
/*@
   FETIScalingSetUp - Computes the scaling for the FETI method

   Input Parameter:
.  ft - the FETI context

   Notes:
   It must be called after calling SubdomainSetUp()

   Level: developer

.keywords: FETI

.seealso: FETICreate(), FETISetUp(), SubdomainSetUp()
@*/
PetscErrorCode  FETIScalingSetUp(FETI ft)
{
  PetscErrorCode ierr,(*func)(FETI);
  Subdomain      sd = ft->subdomain;
  char           type[256];
  const char*    def="scrho";
  const char*    scltype;
  PetscBool      flg;
  MPI_Comm       comm;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr  = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  if(!sd->N_to_B) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: SubdomainSetUp must be first called");
  ierr = PetscOptionsBegin(comm,NULL,"FETI Scaling Options","");CHKERRQ(ierr);
  {
  ierr = PetscOptionsFList("-feti_scaling_type","FETIScaling","FETIScalingSetUp",FETIScalingList,def,type,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-feti_scaling_factor",&ft->scaling_factor,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    scltype = type;
  } else {
    ierr = PetscStrcmp((char*)(ft->scaling_type),SCUNK,&flg);CHKERRQ(ierr);
    scltype = flg ? def : ft->scaling_type;
  }
  ierr = PetscFunctionListFind(FETIScalingList,scltype,&func);CHKERRQ(ierr);    
  if (!func) SETERRQ1(comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested FETIScaling type %s",scltype);
  /* destroy previous scaling vector */
  ierr = VecDestroy(&ft->Wscaling);CHKERRQ(ierr);
  ierr = (*func)(ft);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIScalingDestroy"
PetscErrorCode  FETIScalingDestroy(FETI);
PetscErrorCode  FETIScalingDestroy(FETI feti)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(feti,FETI_CLASSID,1);
  ierr = VecDestroy(&feti->Wscaling);CHKERRQ(ierr);
  feti->scaling_factor = 1.0;
  feti->scaling_type   = SCNONE;
  PetscFunctionReturn(0);
}

