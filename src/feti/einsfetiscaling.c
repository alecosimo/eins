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
  Vec              lambda_local;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr = VecGetArray(sd->vec1_B,&array);CHKERRQ(ierr);
  for ( i=0;i<sd->n_B;i++ ) { array[i]=1.0/(sd->count[i]+1);}
  ierr = VecRestoreArray(sd->vec1_B,&array);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(ft->lambda_global,&lambda_local);CHKERRQ(ierr);
  ierr = MatMult(ft->B_Ddelta,sd->vec1_B,lambda_local);CHKERRQ(ierr);
  /* scaling */
  ierr = MatCopy(ft->B_delta,ft->B_Ddelta,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDiagonalScale(ft->B_Ddelta,lambda_local,NULL);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVector(ft->lambda_global,lambda_local);CHKERRQ(ierr);
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
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  ierr = MatCopy(ft->B_delta,ft->B_Ddelta,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
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
  Vec              lambda_local,lambda_k;
  Subdomain        sd = ft->subdomain;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  PetscValidHeaderSpecific(ft,FETI_CLASSID,1);
  /* Locally get the diagonal */
  ierr = MatGetDiagonal(sd->localA,sd->vec1_N);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_N,sd->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Communicate and get neighbors diagonal */
  ierr = VecUnAsmGetLocalVector(ft->lambda_global,&lambda_local);CHKERRQ(ierr);
  ierr = MatMult(ft->B_Ddelta,sd->vec1_B,lambda_local);CHKERRQ(ierr);
  ierr = VecExchangeBegin(ft->exchange_lambda,ft->lambda_global,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(ft->exchange_lambda,ft->lambda_global,INSERT_VALUES);CHKERRQ(ierr);
  /* compute the summation of diagonals from neighbors, to be stored in sd->vec2_B */
  ierr = MatMultTranspose(ft->B_Ddelta,lambda_local,sd->vec2_B);CHKERRQ(ierr);
  ierr = VecAXPY(sd->vec2_B,1,sd->vec1_B);CHKERRQ(ierr);
  /* point-wise divide lambda */
  ierr = VecDuplicate(lambda_local,&lambda_k);CHKERRQ(ierr);
  ierr = MatMult(ft->B_Ddelta,sd->vec2_B,lambda_k);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(lambda_local,lambda_local,lambda_k);CHKERRQ(ierr);
  /* scaling */
  ierr = MatCopy(ft->B_delta,ft->B_Ddelta,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatDiagonalScale(ft->B_Ddelta,lambda_local,NULL);CHKERRQ(ierr);

  ierr = VecUnAsmRestoreLocalVector(ft->lambda_global,lambda_local);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda_k);CHKERRQ(ierr);
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
  if(!ft->B_delta) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: Constraint matrix B must be first created");
  if(!ft->B_Ddelta) {
    ierr = MatConvert(ft->B_delta,MATSAME,MAT_INITIAL_MATRIX,&ft->B_Ddelta);CHKERRQ(ierr);
  }
  if(!sd->N_to_B) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: SubdomainSetUp must be first called");
  ierr = PetscOptionsBegin(comm,NULL,"FETI Scaling Options","");CHKERRQ(ierr);
  {
  ierr = PetscOptionsFList("-feti_scaling_type","FETIScaling","FETIScalingSetUp",FETIScalingList,def,type,256,&flg);CHKERRQ(ierr);
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
  ierr = (*func)(ft);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIScalingDestroy"
PetscErrorCode  FETIScalingDestroy(FETI);
PetscErrorCode  FETIScalingDestroy(FETI feti)
{  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(feti,FETI_CLASSID,1);
  feti->scaling_type   = SCNONE;
  PetscFunctionReturn(0);
}

