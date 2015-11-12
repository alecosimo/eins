#include <../src/vec/einsvecunasm.h>
#include <petsc/private/matimpl.h>
#include <petscblaslapack.h>

static PetscErrorCode VecView_UNASM(Vec,PetscViewer);
static PetscErrorCode VecDuplicate_UNASM(Vec,Vec*);
static PetscErrorCode VecDestroy_UNASM(Vec);
static PetscErrorCode VecGetLocalSize_UNASM(Vec,PetscInt*);
static PetscErrorCode VecGetSize_UNASM(Vec,PetscInt*);
static PetscErrorCode VecDot_UNASM(Vec,Vec,PetscScalar*);
static PetscErrorCode VecMDot_UNASM(Vec,PetscInt,const Vec[],PetscScalar*);
static PetscErrorCode VecNorm_UNASM(Vec,NormType,PetscReal*);
static PetscErrorCode VecTDot_UNASM(Vec,Vec,PetscScalar*);
static PetscErrorCode VecScale_UNASM(Vec,PetscScalar);
static PetscErrorCode VecCopy_UNASM(Vec,Vec);
static PetscErrorCode VecSet_UNASM(Vec,PetscScalar);
static PetscErrorCode VecAXPY_UNASM(Vec,PetscScalar,Vec);
static PetscErrorCode VecMAXPY_UNASM(Vec,PetscInt,const PetscScalar*,Vec*);
static PetscErrorCode VecAYPX_UNASM(Vec,PetscScalar,Vec);
static PetscErrorCode VecSetValues_UNASM(Vec,PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
static PetscErrorCode VecMTDot_UNASM(Vec,PetscInt,const Vec [],PetscScalar*);
static PetscErrorCode VecAssemblyBegin_UNASM(Vec);
static PetscErrorCode VecAssemblyEnd_UNASM(Vec);
static PetscErrorCode VecGetArrayRead_UNASM(Vec,const PetscScalar**);
static PetscErrorCode VecRestoreArrayRead_UNASM(Vec,const PetscScalar**);
static PetscErrorCode VecGetArray_UNASM(Vec,PetscScalar**);
static PetscErrorCode VecRestoreArray_UNASM(Vec,PetscScalar**);
  
#undef __FUNCT__
#define __FUNCT__ "VecCreate_UNASM"
/*@ VECMPIUNASM = "mpiunasm" - Globally unassembled mpi vector for
   using in FETI-based domain decomposition methods. Each processor
   redundantly owns a portion of the global unassembled vector. A
   given processor can only access its local portion of the global
   vector using the local numbering of thentries. In order to access
   the global vector you have to get an assembled version of the
   vector by calling VecGetAssembledMPIVec().

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VECMPI, VecType, VecGetAssembledMPIVec
@*/
PETSC_EXTERN PetscErrorCode VecCreate_UNASM(Vec v)
{
  PetscErrorCode ierr;
  Vec_UNASM      *b;

  PetscFunctionBegin;
  ierr    = PetscNewLog(v,&b);CHKERRQ(ierr);
  v->data = (void*)b;
  /* create local vec */
  ierr = VecCreateSeq(PETSC_COMM_SELF,v->map->n,&b->vlocal);CHKERRQ(ierr);
  b->multiplicity          = 0;
  /* vector ops */
  ierr                     = PetscMemzero(v->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
  v->ops->duplicate        = VecDuplicate_UNASM;
  v->ops->duplicatevecs    = VecDuplicateVecs_Default;      
  v->ops->getlocalsize     = VecGetLocalSize_UNASM;
  v->ops->getsize          = VecGetSize_UNASM;
  v->ops->dot              = VecDot_UNASM;
  v->ops->scale            = VecScale_UNASM;
  v->ops->mdot             = VecMDot_UNASM;
  v->ops->norm             = VecNorm_UNASM;
  v->ops->tdot             = VecTDot_UNASM;
  v->ops->mtdot            = VecMTDot_UNASM;
  v->ops->copy             = VecCopy_UNASM;
  v->ops->set              = VecSet_UNASM;
  v->ops->axpy             = VecAXPY_UNASM;
  v->ops->maxpy            = VecMAXPY_UNASM;
  v->ops->aypx             = VecAYPX_UNASM;
  v->ops->setvalues        = VecSetValues_UNASM;
  v->ops->assemblybegin    = VecAssemblyBegin_UNASM;
  v->ops->assemblyend      = VecAssemblyEnd_UNASM;
  v->ops->destroy          = VecDestroy_UNASM;
  v->ops->view             = VecView_UNASM;
  v->ops->getarrayread     = VecGetArrayRead_UNASM;
  v->ops->getarray         = VecGetArray_UNASM;
  v->ops->restorearrayread = VecRestoreArrayRead_UNASM;
  v->ops->restorearray     = VecRestoreArray_UNASM;
    
  ierr = PetscObjectChangeTypeName((PetscObject)v,VECMPIUNASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSetValues_UNASM"
static PetscErrorCode VecSetValues_UNASM(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode m)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  PetscScalar    *xx;
  PetscInt       i;
  Vec            pxin;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pxin = xi->vlocal;
  ierr = VecGetArray(pxin,&xx);CHKERRQ(ierr);
  if (m == INSERT_VALUES) {
    for (i=0; i<ni; i++) {
      if (pxin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (ix[i] < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if (ix[i] >= pxin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->n);
#endif
      xx[ix[i]] = y[i];
    }
  } else {
    for (i=0; i<ni; i++) {
      if (pxin->stash.ignorenegidx && ix[i] < 0) continue;
#if defined(PETSC_USE_DEBUG)
      if (ix[i] < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D cannot be negative",ix[i]);
      if (ix[i] >= pxin->map->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %D maximum %D",ix[i],xin->map->n);
#endif
      xx[ix[i]] += y[i];
    }
  }
  ierr = VecRestoreArray(pxin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecUnAsmGetLocalVector"
/*@ VecUnAsmGetLocalVector - Gets a reference to the local vector of
  the given globally unassembled vector WITHOUT incrementing the
  reference count of the object, so the obtained reference to the
  vector must not be destroyed by the user.

  Input Parameter:
.  xin - the globally unassembled vector.

  Output Parameter:
.  vec - reference to the local vector.

  Level: intermediate

@*/
PETSC_EXTERN PetscErrorCode VecUnAsmGetLocalVector(Vec xin,Vec *vec)
{
  Vec_UNASM      *xi; 
  PetscBool      flg;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(xin,VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)xin,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Cannot get local vector from non globally unassembled vector");
  xi = (Vec_UNASM*)xin->data;
  *vec = xi->vlocal;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecMAXPY_UNASM"
static PetscErrorCode VecMAXPY_UNASM(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data,*yi;
  Vec            *y_aux;
  PetscInt       i;
  
  PetscFunctionBegin;
  ierr = PetscMalloc1(nv,&y_aux);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    yi       = (Vec_UNASM*)y[i]->data;
    y_aux[i] = yi->vlocal;
  } 
  ierr = VecMAXPY_Seq(xi->vlocal,nv,alpha,y_aux);CHKERRQ(ierr);
  ierr = PetscFree(y_aux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGetArrayRead_UNASM"
static PetscErrorCode VecGetArrayRead_UNASM(Vec x,const PetscScalar **a)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)x->data;
  
  PetscFunctionBegin;
  ierr = VecGetArrayRead(xi->vlocal,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecRestoreArrayRead_UNASM"
static PetscErrorCode VecRestoreArrayRead_UNASM(Vec x,const PetscScalar **a)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)x->data;
  
  PetscFunctionBegin;
  ierr = VecRestoreArrayRead(xi->vlocal,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGetArray_UNASM"
static PetscErrorCode VecGetArray_UNASM(Vec x,PetscScalar **a)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)x->data;
  
  PetscFunctionBegin;
  ierr = VecGetArray(xi->vlocal,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecRestoreArray_UNASM"
static PetscErrorCode VecRestoreArray_UNASM(Vec x,PetscScalar **a)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)x->data;
  
  PetscFunctionBegin;
  ierr = VecRestoreArray(xi->vlocal,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecAYPX_UNASM"
static PetscErrorCode VecAYPX_UNASM(Vec yin,PetscScalar alpha,Vec xin)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec_UNASM      *yi = (Vec_UNASM*)yin->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecAYPX_Seq(yi->vlocal,alpha,xi->vlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}


#undef __FUNCT__
#define __FUNCT__ "VecAXPY_UNASM"
static PetscErrorCode VecAXPY_UNASM(Vec yin,PetscScalar alpha,Vec xin)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec_UNASM      *yi = (Vec_UNASM*)yin->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecAXPY_Seq(yi->vlocal,alpha,xi->vlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}


#undef __FUNCT__
#define __FUNCT__ "VecSet_UNASM"
static PetscErrorCode VecSet_UNASM(Vec xin,PetscScalar alpha)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet_Seq(xi->vlocal,alpha);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecAssemblyEnd_UNASM"
static PetscErrorCode VecAssemblyEnd_UNASM(Vec xin)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecAssemblyBegin(xi->vlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecAssemblyBegin_UNASM"
static PetscErrorCode VecAssemblyBegin_UNASM(Vec xin)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecAssemblyBegin(xi->vlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecDot_UNASM"
static PetscErrorCode VecDot_UNASM(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  Vec            mp;
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec_UNASM      *yi = (Vec_UNASM*)yin->data;

  PetscFunctionBegin;
  if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
  ierr = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr);
  ierr = VecDot_Seq(mp,yi->vlocal,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  ierr = VecDestroy(&mp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecUnAsmSetMultiplicity"
/*@ VecUnAsmSetMultiplicity - This function is specific of MPI
  globally unassebled vectors. It sets the multiplicity of the entries
  of the local vector owned by this processor. It is considered that
  the multiplicity is equal to the number of times that the given
  entry is shared between processors in the communicator of the
  associated globally unassembled vector.

   Input Parameters:
.  x             - the global unassembled vector
.  multiplicity  - vector with the multiplicity of the entries of the local vector owned by this processor.

   Level: beginner

.seealso VECMPIUNASM
@*/
PETSC_EXTERN PetscErrorCode VecUnAsmSetMultiplicity(Vec x,Vec multiplicity)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi;
  PetscBool      flg;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(multiplicity,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)x,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot set multiplicity to non globally unassembled vector");

  xi = (Vec_UNASM*)x->data;
  xi->multiplicity = multiplicity;
  ierr             = PetscObjectReference((PetscObject)multiplicity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecMDot_UNASM"
static PetscErrorCode VecMDot_UNASM(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec            *y_aux,mp;
  Vec_UNASM      *yi;
  PetscInt       i;
  
  PetscFunctionBegin;
  if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
  ierr = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr); 
  ierr = PetscMalloc1(nv,&y_aux);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    yi       = (Vec_UNASM*)y[i]->data;
    y_aux[i] = yi->vlocal;
  }
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_Seq(mp,nv,y_aux,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&mp);CHKERRQ(ierr);
  ierr = PetscFree(y_aux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecTDot_UNASM"
static PetscErrorCode VecTDot_UNASM(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec_UNASM      *yi = (Vec_UNASM*)yin->data;
  Vec            mp;
  
  PetscFunctionBegin;
  if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
  ierr = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr);
  ierr = VecTDot_Seq(mp,yi->vlocal,&work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  ierr = VecDestroy(&mp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecMTDot_UNASM"
static PetscErrorCode VecMTDot_UNASM(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec            *y_aux,mp;
  Vec_UNASM      *yi;
  PetscInt       i;
  
  PetscFunctionBegin;
  if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
  ierr = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr); 
  ierr = PetscMalloc1(nv,&y_aux);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    yi       = (Vec_UNASM*)y[i]->data;
    y_aux[i] = yi->vlocal;
  }
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMTDot_Seq(mp,nv,y_aux,work);CHKERRQ(ierr);
  ierr = MPI_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&mp);CHKERRQ(ierr);
  ierr = PetscFree(y_aux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecScale_UNASM"
static PetscErrorCode VecScale_UNASM(Vec xin, PetscScalar alpha)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  
  PetscFunctionBegin;
  ierr = VecScale_Seq(xi->vlocal,alpha);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecNorm_UNASM"
static PetscErrorCode VecNorm_UNASM(Vec xin,NormType type,PetscReal *z)
{
  PetscReal         sum,work = 0.0;
  const PetscScalar *xx,*yy;
  PetscErrorCode    ierr;
  PetscInt          n   = xin->map->n;
  PetscBLASInt      one = 1,bn;
  Vec_UNASM         *xi = (Vec_UNASM*)xin->data;
  Vec               mp;
  
  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
    ierr = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr);
    ierr = VecGetArrayRead(mp,&xx);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xi->vlocal,&yy);CHKERRQ(ierr);
    work = PetscRealPart(BLASdot_(&bn,xx,&one,yy,&one));
    ierr = VecRestoreArrayRead(mp,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xi->vlocal,&yy);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    ierr = VecDestroy(&mp);CHKERRQ(ierr);
    *z   = PetscSqrtReal(sum);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
    ierr = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr);
    ierr = VecNorm_Seq(mp,NORM_1,&work);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    ierr = VecDestroy(&mp);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    ierr = VecNorm_Seq(xi->vlocal,NORM_INFINITY,&work);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
    {
      PetscReal temp[2];
      ierr    = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
      ierr    = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr);
      ierr    = VecGetArrayRead(mp,&xx);CHKERRQ(ierr);
      ierr    = VecGetArrayRead(xi->vlocal,&yy);CHKERRQ(ierr);
      temp[1] = PetscRealPart(BLASdot_(&bn,xx,&one,yy,&one));
      ierr    = VecRestoreArrayRead(mp,&xx);CHKERRQ(ierr);
      ierr    = VecRestoreArrayRead(xi->vlocal,&yy);CHKERRQ(ierr);
      ierr    = VecNorm_Seq(mp,NORM_1,temp);CHKERRQ(ierr);
      ierr    = MPI_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
      z[1]    = PetscSqrtReal(z[1]);
      ierr    = VecDestroy(&mp);CHKERRQ(ierr);
      ierr    = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecDuplicate_UNASM"
static PetscErrorCode VecDuplicate_UNASM(Vec win,Vec *V)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi;
  
  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),V);CHKERRQ(ierr);
  ierr = PetscObjectSetPrecision((PetscObject)*V,((PetscObject)win)->precision);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,win->map->n,win->map->N);CHKERRQ(ierr);
  ierr = VecSetType(*V,((PetscObject)win)->type_name);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);
  xi   = (Vec_UNASM*)win->data;
  if(xi->multiplicity) {ierr = VecUnAsmSetMultiplicity(*V,xi->multiplicity);CHKERRQ(ierr);}
    
  (*V)->ops->view          = win->ops->view;
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx; 
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCopy_UNASM"
static PetscErrorCode VecCopy_UNASM(Vec xin,Vec yin)
{
  PetscErrorCode    ierr;
  Vec_UNASM         *xi = (Vec_UNASM*)xin->data;
  Vec_UNASM         *yi = (Vec_UNASM*)yin->data;
  
  PetscFunctionBegin;
  ierr = VecCopy(xi->vlocal,yi->vlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGetSize_UNASM"
static PetscErrorCode VecGetSize_UNASM(Vec v,PetscInt* s)
{
  PetscFunctionBegin;
  *s = v->map->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy_UNASM"
static PetscErrorCode VecDestroy_UNASM(Vec v)
{
  PetscErrorCode ierr;
  Vec_UNASM      *b = (Vec_UNASM*)v->data;
 
  PetscFunctionBegin;
  ierr = VecDestroy(&b->vlocal);CHKERRQ(ierr);
  ierr = VecDestroy(&b->multiplicity);CHKERRQ(ierr);
  ierr = PetscFree(v->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecGetLocalSize_UNASM"
static PetscErrorCode VecGetLocalSize_UNASM(Vec v,PetscInt* s)
{
  Vec_UNASM      *b = (Vec_UNASM*)v->data;

  PetscFunctionBegin;
  *s = (b->vlocal)->map->n;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecUnAsmCreateMPIVec"
/*@
   VecUnAsmCreateMPIVec - Creates an MPI distributed vector by assembling the given globally unassembled vector.

   Input Parameter:
.  v           - The globally unassembled vector.
.  compat      - Rule to use in order to impose compatibility of shared DOFs.
.  mapping     - The local to global mapping for the vector entries.

   Output Parameter:
.  _vec         - The created distributed vector.

   Level: intermediate

.seealso CompatibilityRule
@*/
PetscErrorCode VecUnAsmCreateMPIVec(Vec v,ISLocalToGlobalMapping mapping,CompatibilityRule compat,Vec *_vec)
{
  Vec            mpivec;
  PetscErrorCode ierr;
  Vec_UNASM      *xi;
  PetscBool      flg;
  
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)v,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_SUP,"Cannot create MPI vector from non globally unassembled vector");
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,2);

  xi    = (Vec_UNASM*)v->data;
  ierr  = VecCreate(PetscObjectComm((PetscObject)v),&mpivec);CHKERRQ(ierr);
  ierr  = VecSetType(mpivec,VECMPI);CHKERRQ(ierr);
  ierr  = VecSetSizes(mpivec,PETSC_DECIDE,v->map->N);CHKERRQ(ierr);
  ierr  = VecSetLocalToGlobalMapping(mpivec,mapping);CHKERRQ(ierr);

  if (compat == COMPAT_RULE_AVG) {
    if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
    {
      PetscInt          *idx,i;
      const PetscScalar *xx;
      Vec               mp;
      ierr = VecDuplicate(xi->vlocal,&mp);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(mp,xi->vlocal,xi->multiplicity);CHKERRQ(ierr);
      ierr = PetscMalloc1(v->map->n,&idx);CHKERRQ(ierr);
      for(i=0;i<v->map->n;i++) idx[i] = i;
      ierr = VecGetArrayRead(mp,&xx);CHKERRQ(ierr);
      ierr = VecSet(mpivec,0.0);CHKERRQ(ierr);
      ierr = VecSetValuesLocal(mpivec,v->map->n,idx,xx,ADD_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(mpivec);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(mpivec);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(mp,&xx);CHKERRQ(ierr);
      ierr = VecDestroy(&mp);CHKERRQ(ierr);
    }
  }
  *_vec = mpivec;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecView_UNASM"
static PetscErrorCode VecView_UNASM(Vec xin,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  PetscMPIInt    size,rank,buff=1;
  MPI_Status     status;
  MPI_Comm       comm;
  PetscBool      iascii;
  
  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);
  ierr = MPI_Comm_rank(comm,&rank);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);

  if(iascii){
    if(rank) { ierr = MPI_Recv(&buff,1,MPI_INT,rank-1,0,comm,&status);CHKERRQ(ierr); }
    PetscPrintf(PETSC_COMM_SELF, "Processor # %d, size %d \n",rank,size);
    ierr = VecView(xi->vlocal,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = MPI_Send(&buff,1,MPI_INT,(rank+1)%size,0,comm);CHKERRQ(ierr);
    if(!rank) { ierr = MPI_Recv(&buff,1,MPI_INT,size-1,0,comm,&status);CHKERRQ(ierr); }
  } else {
    SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Error: only ascii viewer implemented");
  }
  
  PetscFunctionReturn(0);
}


