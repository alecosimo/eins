#include <../src/vec/einsvecunasm.h>
#include <petsc/private/matimpl.h>
#include <petscblaslapack.h>
#include <petscviewerhdf5.h>

static PetscErrorCode VecCreate_UNASM_Private(Vec,const PetscScalar[],Vec);
static PetscErrorCode VecView_UNASM(Vec,PetscViewer);
static PetscErrorCode VecDuplicate_UNASM(Vec,Vec*);
static PetscErrorCode VecDestroy_UNASM(Vec);
static PetscErrorCode VecGetLocalSize_UNASM(Vec,PetscInt*);
static PetscErrorCode VecAXPBY_UNASM(Vec,PetscScalar,PetscScalar,Vec);
static PetscErrorCode VecGetSize_UNASM(Vec,PetscInt*);
static PetscErrorCode VecDot_UNASM(Vec,Vec,PetscScalar*);
static PetscErrorCode VecMDot_UNASM(Vec,PetscInt,const Vec[],PetscScalar*);
static PetscErrorCode VecNorm_UNASM(Vec,NormType,PetscReal*);
static PetscErrorCode VecTDot_UNASM(Vec,Vec,PetscScalar*);
static PetscErrorCode VecScale_UNASM(Vec,PetscScalar);
static PetscErrorCode VecCopy_UNASM(Vec,Vec);
static PetscErrorCode VecSet_UNASM(Vec,PetscScalar);
static PetscErrorCode VecAXPY_UNASM(Vec,PetscScalar,Vec);
static PetscErrorCode VecWAXPY_UNASM(Vec,PetscScalar,Vec,Vec);
static PetscErrorCode VecMAXPY_UNASM(Vec,PetscInt,const PetscScalar*,Vec*);
static PetscErrorCode VecAYPX_UNASM(Vec,PetscScalar,Vec);
static PetscErrorCode VecSetValuesLocal_UNASM(Vec,PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
static PetscErrorCode VecMTDot_UNASM(Vec,PetscInt,const Vec [],PetscScalar*);
static PetscErrorCode VecAssemblyBegin_UNASM(Vec);
static PetscErrorCode VecAssemblyEnd_UNASM(Vec);
static PetscErrorCode VecGetArrayRead_UNASM(Vec,const PetscScalar**);
static PetscErrorCode VecRestoreArrayRead_UNASM(Vec,const PetscScalar**);
static PetscErrorCode VecGetArray_UNASM(Vec,PetscScalar**);
static PetscErrorCode VecRestoreArray_UNASM(Vec,PetscScalar**);
#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode VecView_UNASM_HDF5(Vec,PetscViewer);
#endif


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
  
  PetscFunctionBegin;
  ierr = VecCreate_UNASM_Private(v,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "VecCreate_UNASM_Private"
/*@ VecCreate_UNASM_Private - Creates and intializes globally
  unassembled vector. It is called from VecCreate_UNASM and
  VecCreateMPIUnasmWithArray.

   Input Parameters:
.  array   - array used to set create the local vector corresponding to each subdomain
.  v       - the context of the vector to be created

  Level: developer

.seealso: VecCreate(), VecCreateMPIUnasmWithArray()
@*/
static PetscErrorCode VecCreate_UNASM_Private(Vec v,const PetscScalar array[],Vec localVec)
{
  PetscErrorCode ierr;
  Vec_UNASM      *b;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  char           str[10];
  
  PetscFunctionBegin;
  ierr    = PetscNewLog(v,&b);CHKERRQ(ierr);
  v->data = (void*)b;
  /* create local vec */
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);
  if((v->map->n) == PETSC_DECIDE) { SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot set local size to PETSC_DECIDE"); }
  if(!localVec) {
    if(!array) {
      ierr = VecCreateSeq(PETSC_COMM_SELF,v->map->n,&b->vlocal);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,v->map->n,array,&b->vlocal);CHKERRQ(ierr);
    }
  } else {
    b->vlocal = localVec;
    ierr = PetscObjectReference((PetscObject)localVec);CHKERRQ(ierr);
  }
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  sprintf(str,NAMEDOMAIN,rank);
  ierr = PetscObjectSetName((PetscObject)b->vlocal,str);CHKERRQ(ierr);
  if((v->map->N) == PETSC_DECIDE) { ierr = MPI_Allreduce(&v->map->n,&v->map->N,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr); }
  
  b->multiplicity          = 0;
  b->local_sizes           = 0;
  b->feti                  = 0;
  /* vector ops */
  ierr                     = PetscMemzero(v->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
  v->ops->duplicate        = VecDuplicate_UNASM;
  v->ops->duplicatevecs    = VecDuplicateVecs_Default;
  v->ops->destroyvecs      = VecDestroyVecs_Default;      
  v->ops->getlocalsize     = VecGetLocalSize_UNASM;
  v->ops->getsize          = VecGetSize_UNASM;
  v->ops->dot              = VecDot_UNASM;
  v->ops->scale            = VecScale_UNASM;
  v->ops->mdot             = VecMDot_UNASM;
  v->ops->norm             = VecNorm_UNASM;
  v->ops->tdot             = VecTDot_UNASM;
  v->ops->mtdot            = VecMTDot_UNASM;
  v->ops->copy             = VecCopy_UNASM;
  v->ops->norm_local       = VecNorm_Seq; 
  v->ops->set              = VecSet_UNASM;
  v->ops->axpy             = VecAXPY_UNASM;
  v->ops->axpby            = VecAXPBY_UNASM;
  v->ops->waxpy            = VecWAXPY_UNASM;
  v->ops->maxpy            = VecMAXPY_UNASM;
  v->ops->aypx             = VecAYPX_UNASM;
  v->ops->setvalueslocal   = VecSetValuesLocal_UNASM;
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
#define __FUNCT__ "VecSetValuesLocal_UNASM"
static PetscErrorCode VecSetValuesLocal_UNASM(Vec xin,PetscInt ni,const PetscInt ix[],const PetscScalar y[],InsertMode m)
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
#define __FUNCT__ "VecScatterUABegin"
/*@ VecScatterUABegin - Performs a VecScatter with one or both of the intervenient 
  vectors of type MPIUNASM. The layout of the local vectors comprising the considered 
  globally unassembled vectors must matched the layout of the vectors used for 
  creating the VecScatter. 

  Input Parameter:
.  vecscatter  -  The VecScatter context.
.  v1          -  Vector of origin for performing the scatter.
.  v2          -  Vector of destination for performing the scatter.
.  imode       -  Insert mode
.  smode       -  Scatter mode

  Level: basic

@*/
PETSC_EXTERN PetscErrorCode VecScatterUABegin(VecScatter vecscatter,Vec v1,Vec v2,InsertMode imode,ScatterMode smode)
{
  PetscBool      flg;
  Vec            va1,va2;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(v2,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)v1,VECMPIUNASM,&flg);CHKERRQ(ierr);
  va1  = (flg)? ((Vec_UNASM*)v1->data)->vlocal: v1;
  ierr = PetscObjectTypeCompare((PetscObject)v2,VECMPIUNASM,&flg);CHKERRQ(ierr);
  va2  = (flg)? ((Vec_UNASM*)v2->data)->vlocal: v2;
  ierr = VecScatterBegin(vecscatter,va1,va2,imode,smode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecScatterUAEnd"
/*@ VecScatterUAEnd - Performs a VecScatter with one or both of the intervenient 
  vectors of type MPIUNASM. The layout of the local vectors comprising the considered 
  globally unassembled vectors must matched the layout of the vectors used for 
  creating the VecScatter. 

  Input Parameter:
.  vecscatter  -  The VecScatter context.
.  v1          -  Vector of origin for performing the scatter.
.  v2          -  Vector of destination for performing the scatter.
.  imode       -  Insert mode
.  smode       -  Scatter mode

  Level: basic

@*/
PETSC_EXTERN PetscErrorCode VecScatterUAEnd(VecScatter vecscatter,Vec v1,Vec v2,InsertMode imode,ScatterMode smode)
{
  PetscBool      flg;
  Vec            va1,va2;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(v2,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)v1,VECMPIUNASM,&flg);CHKERRQ(ierr);
  va1  = (flg)? ((Vec_UNASM*)v1->data)->vlocal: v1;
  ierr = PetscObjectTypeCompare((PetscObject)v2,VECMPIUNASM,&flg);CHKERRQ(ierr);
  va2  = (flg)? ((Vec_UNASM*)v2->data)->vlocal: v2;
  ierr = VecScatterEnd(vecscatter,va1,va2,imode,smode);CHKERRQ(ierr);
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
#define __FUNCT__ "VecWAXPY_UNASM"
static PetscErrorCode VecWAXPY_UNASM(Vec win,PetscScalar alpha,Vec xin,Vec yin)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec_UNASM      *yi = (Vec_UNASM*)yin->data;
  Vec_UNASM      *wi = (Vec_UNASM*)win->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecWAXPY_Seq(wi->vlocal,alpha,xi->vlocal,yi->vlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}


#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_UNASM"
static PetscErrorCode VecAXPBY_UNASM(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  Vec_UNASM      *xi = (Vec_UNASM*)xin->data;
  Vec_UNASM      *yi = (Vec_UNASM*)yin->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecAXPBY_Seq(yi->vlocal,alpha,beta,xi->vlocal);CHKERRQ(ierr);
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
#define __FUNCT__ "VecGetFETI"
/*@ VecGetFETI - Gets FETI context assigned to a globally unassembled vector.

   Input Parameters:
+  x     - the global unassembled vector

   Output Parameters:
+  feti  - the feti context

   Level: beginner

.seealso VECMPIUNASM
@*/
PETSC_EXTERN PetscErrorCode VecGetFETI(Vec x,FETI *feti)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi;
  PetscBool      flg;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)x,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot get feti context from non globally unassembled vector");

  xi       = (Vec_UNASM*)x->data;
  *feti    = xi->feti;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSetFETI"
/*@ VecSetFETI - Sets FETI context to a globally unassembled vector.

   Input Parameters:
+  x     - the global unassembled vector
-  feti  - the feti context

   Level: beginner

.seealso VECMPIUNASM
@*/
PETSC_EXTERN PetscErrorCode VecSetFETI(Vec x,FETI feti)
{
  PetscErrorCode ierr;
  Vec_UNASM      *xi;
  PetscBool      flg;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,1);
  PetscValidHeaderSpecific(feti,FETI_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)x,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot set feti context to non globally unassembled vector");

  xi       = (Vec_UNASM*)x->data;
  xi->feti = feti;
  ierr     = PetscObjectReference((PetscObject)feti);CHKERRQ(ierr);
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
  if(xi->feti) {
    ierr = FETIComputeForceNorm(xi->feti,xin,type,z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } 
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
  ierr = FETIDestroy(&b->feti);CHKERRQ(ierr);
  if(b->local_sizes) { ierr = PetscFree(b->local_sizes);CHKERRQ(ierr);}
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
  PetscErrorCode    ierr;
  Vec_UNASM         *xi = (Vec_UNASM*)xin->data;
  PetscMPIInt       j,n,size,rank,tag;
  PetscInt          work = xin->map->n,len;
  MPI_Status        status;
  MPI_Comm          comm;
  PetscBool         iascii;
  Vec               vec;
  PetscScalar       *values;
  const PetscScalar *xarray;
  char              str[10];
#if defined(PETSC_HAVE_HDF5)
  PetscBool      ishdf5;
#endif
  
  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);
  ierr = MPI_Comm_rank(comm,&rank);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
#endif

  if(iascii){
    ierr = MPI_Reduce(&work,&len,1,MPIU_INT,MPI_MAX,0,comm);CHKERRQ(ierr);
    tag  = ((PetscObject)viewer)->tag;
    if(!rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "Processor # %d out of %d \n",rank,size);CHKERRQ(ierr);
      ierr = VecView(xi->vlocal,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      ierr = PetscMalloc1(len,&values);CHKERRQ(ierr);
      for (j=1; j<size; j++) {
	ierr = MPI_Recv(values,(PetscMPIInt)len,MPIU_SCALAR,j,tag,comm,&status);CHKERRQ(ierr);
        ierr = MPI_Get_count(&status,MPIU_SCALAR,&n);CHKERRQ(ierr);
	ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,values,&vec);CHKERRQ(ierr);
	sprintf(str,NAMEDOMAIN,j);
	ierr = PetscObjectSetName((PetscObject)vec,str);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_SELF, "Processor # %d out of %d \n",j,size);CHKERRQ(ierr);
	ierr = VecView(vec,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
	ierr = VecDestroy(&vec);CHKERRQ(ierr);
      }
      ierr = PetscFree(values);CHKERRQ(ierr);
    } else {
      ierr = VecGetArrayRead(xi->vlocal,&xarray);CHKERRQ(ierr);
      ierr = MPI_Send((void*)xarray,xin->map->n,MPIU_SCALAR,0,tag,comm);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(xi->vlocal,&xarray);CHKERRQ(ierr);
    }    
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = VecView_UNASM_HDF5(xin,viewer);CHKERRQ(ierr);
#endif
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Not supported viewer");
  }
  
  PetscFunctionReturn(0);
}


#if defined(PETSC_HAVE_HDF5)
#undef __FUNCT__
#define __FUNCT__ "VecView_UNASM_HDF5"
static PetscErrorCode VecView_UNASM_HDF5(Vec xin, PetscViewer viewer)
{
  hid_t             filespace; /* file dataspace identifier */
  hid_t             chunkspace; /* chunk dataset property identifier */
  hid_t             plist_id;  /* property list identifier */
  hid_t             dset_id;   /* dataset identifier */
  hid_t             memspace;  /* memory dataspace identifier */
  hid_t             file_id;
  hid_t             groupS,group;
  hid_t             memscalartype; /* scalar type for mem (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  hid_t             filescalartype; /* scalar type for file (H5T_NATIVE_FLOAT or H5T_NATIVE_DOUBLE) */
  PetscInt          bs = PetscAbs(xin->map->bs);
  hsize_t           dim;
  hsize_t           maxDims[4], dims[4], chunkDims[4], count[4],offset[4];
  PetscInt          timestep, i;
  Vec_UNASM         *xi = (Vec_UNASM*)xin->data;
  const PetscScalar *x;
  const char        *vecname;
  PetscErrorCode    ierr;
  PetscBool         dim2;
  PetscBool         spoutput, found;
  char              vecname_local[10];
  PetscMPIInt       size,rank;
  MPI_Comm          comm;
  
  PetscFunctionBegin;
  ierr = PetscViewerHDF5OpenGroup(viewer, &file_id, &groupS);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) xin, &vecname);CHKERRQ(ierr);
  PetscStackCall("H5Lexists",found = H5Lexists(groupS, vecname, H5P_DEFAULT));
  if (found <= 0) {
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
    PetscStackCallHDF5Return(group,H5Gcreate2,(groupS, vecname, 0, H5P_DEFAULT, H5P_DEFAULT));
#else
    PetscStackCallHDF5Return(group,H5Gcreate,(groupS, vecname, 0));
#endif
    PetscStackCallHDF5(H5Gclose,(group));
  }
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
  PetscStackCallHDF5Return(group,H5Gopen2,(groupS, vecname, H5P_DEFAULT));
#else
  PetscStackCallHDF5Return(group,H5Gopen,groupS, vecname);
#endif 
  PetscStackCallHDF5(H5Gclose,(groupS));

  ierr = PetscViewerHDF5GetTimestep(viewer, &timestep);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetBaseDimension2(viewer,&dim2);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetSPOutput(viewer,&spoutput);CHKERRQ(ierr);

#if defined(PETSC_USE_REAL_SINGLE)
  memscalartype = H5T_NATIVE_FLOAT;
  filescalartype = H5T_NATIVE_FLOAT;
#elif defined(PETSC_USE_REAL___FLOAT128)
#error "HDF5 output with 128 bit floats not supported."
#else
  memscalartype = H5T_NATIVE_DOUBLE;
  if (spoutput == PETSC_TRUE) filescalartype = H5T_NATIVE_FLOAT;
  else filescalartype = H5T_NATIVE_DOUBLE;
#endif

  /* Create the dataset with default properties and close filespace */
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!xi->local_sizes) {
    ierr = PetscMalloc1(size,&xi->local_sizes);CHKERRQ(ierr);
    ierr = MPI_Allgather(&xin->map->n,1,MPIU_INT,xi->local_sizes,1,MPIU_INT,comm);CHKERRQ(ierr);
  }
    /* Create the dataspace for the dataset.
     *
     * dims - holds the current dimensions of the dataset
     *
     * maxDims - holds the maximum dimensions of the dataset (unlimited
     * for the number of time steps with the current dimensions for the
     * other dimensions; so only additional time steps can be added).
     *
     * chunkDims - holds the size of a single time step (required to
     * permit extending dataset).
     */

  for (i=0;i<size;i++) {
    
    sprintf(vecname_local,NAMEDOMAIN,i);

    dim = 0;
    if (timestep >= 0) {
      dims[dim]      = timestep+1;
      maxDims[dim]   = H5S_UNLIMITED;
      chunkDims[dim] = 1;
      ++dim;
    }
    ierr = PetscHDF5IntCast(xi->local_sizes[i],dims + dim);CHKERRQ(ierr);

    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    ++dim;
    if (bs > 1 || dim2) {
      dims[dim]      = bs;
      maxDims[dim]   = dims[dim];
      chunkDims[dim] = dims[dim];
      ++dim;
    }
#if defined(PETSC_USE_COMPLEX)
    dims[dim]      = 2;
    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    ++dim;
#endif

    if (!H5Lexists(group, vecname_local, H5P_DEFAULT)) {
      PetscStackCallHDF5Return(filespace,H5Screate_simple,((int)dim, dims, maxDims));

      /* Create chunk */
      PetscStackCallHDF5Return(chunkspace,H5Pcreate,(H5P_DATASET_CREATE));
      PetscStackCallHDF5(H5Pset_chunk,(chunkspace, (int)dim, chunkDims));

#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
      PetscStackCallHDF5Return(dset_id,H5Dcreate2,(group, vecname_local, filescalartype, filespace, H5P_DEFAULT, chunkspace, H5P_DEFAULT));
#else
      PetscStackCallHDF5Return(dset_id,H5Dcreate,(group, vecname_local, filescalartype, filespace, H5P_DEFAULT));
#endif
      PetscStackCallHDF5(H5Pclose,(chunkspace));
      PetscStackCallHDF5(H5Sclose,(filespace));
    } else {
      PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, vecname_local, H5P_DEFAULT));
      PetscStackCallHDF5(H5Dset_extent,(dset_id, dims));
    }
    PetscStackCallHDF5(H5Dclose,(dset_id));
  }

  /* write data to HDF5 file */
  sprintf(vecname_local,NAMEDOMAIN,rank);
  PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, vecname_local, H5P_DEFAULT));

  /* Each process defines a dataset and writes it to the hyperslab in the file */
  dim = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  ierr = PetscHDF5IntCast(xin->map->n,count + dim);CHKERRQ(ierr);
  ++dim;
  if (bs > 1 || dim2) {
    count[dim] = bs;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  count[dim] = 2;
  ++dim;
#endif
  if (xin->map->n > 0) {
    PetscStackCallHDF5Return(memspace,H5Screate_simple,((int)dim, count, NULL));
  } else {
    /* Can't create dataspace with zero for any dimension, so create null dataspace. */
    PetscStackCallHDF5Return(memspace,H5Screate,(H5S_NULL));
  }

  /* Select hyperslab in the file */
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  ierr = PetscHDF5IntCast(0,offset + dim);CHKERRQ(ierr);
  ++dim;
  if (bs > 1 || dim2) {
    offset[dim] = 0;
    ++dim;
  }
#if defined(PETSC_USE_COMPLEX)
  offset[dim] = 0;
  ++dim;
#endif
  if (xin->map->n > 0) {
    PetscStackCallHDF5Return(filespace,H5Dget_space,(dset_id));
    PetscStackCallHDF5(H5Sselect_hyperslab,(filespace, H5S_SELECT_SET, offset, NULL, count, NULL));
  } else {
    /* Create null filespace to match null memspace. */
    PetscStackCallHDF5Return(filespace,H5Screate,(H5S_NULL));
  }

  /* Create property list for collective dataset write */
  PetscStackCallHDF5Return(plist_id,H5Pcreate,(H5P_DATASET_XFER));
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  PetscStackCallHDF5(H5Pset_dxpl_mpio,(plist_id, H5FD_MPIO_INDEPENDENT));
#endif

  ierr   = VecGetArrayRead(xi->vlocal, &x);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Dwrite,(dset_id, memscalartype, memspace, filespace, plist_id, x));
  PetscStackCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL));
  ierr   = VecRestoreArrayRead(xin, &x);CHKERRQ(ierr);

  /* Close/release resources */
  if (group != file_id) {
    PetscStackCallHDF5(H5Gclose,(group));
  }
  PetscStackCallHDF5(H5Pclose,(plist_id));
  PetscStackCallHDF5(H5Sclose,(filespace)); 
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));
  ierr   = PetscInfo1(xin,"Wrote Vec object with name %s\n",vecname);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#endif


#undef __FUNCT__
#define __FUNCT__ "VecCreateMPIUnasmWithLocalVec"
/*@ VecCreateMPIUnasmWithLocalVec - Creates a globally unassembled vector
  using the provided local vector.

   Input Parameters:
.  comm     - the MPI communicator
.  n        - the local size of the vector
.  N        - the global size of the vector
.  localVec - local vector corresponding to each subdomain

   Output Parameters:
.  V       - the vector to be created

   Level: beginner

.seealso VECMPIUNASM
@*/
PETSC_EXTERN PetscErrorCode VecCreateMPIUnasmWithLocalVec(MPI_Comm comm,PetscInt n,PetscInt N,Vec localVec,Vec *V)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,N);CHKERRQ(ierr); 
  ierr = VecCreate_UNASM_Private(*V,NULL,localVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCreateMPIUnasmWithArray"
/*@ VecCreateMPIUnasmWithArray - Creates a globally unassembled vector
  using the provided array for creating the local vector.

   Input Parameters:
.  comm    - the MPI communicator
.  n       - the local size of the vector
.  N       - the global size of the vector
.  array   - array used to set create the local vector corresponding to each subdomain

   Output Parameters:
.  V       - the vector to be created

   Level: beginner

.seealso VECMPIUNASM
@*/
PETSC_EXTERN PetscErrorCode VecCreateMPIUnasmWithArray(MPI_Comm comm,PetscInt n,PetscInt N,const PetscScalar array[],Vec *V)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,N);CHKERRQ(ierr); 
  ierr = VecCreate_UNASM_Private(*V,array,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecUASum"
/*@
   VecUASum - Computes the sum of all the components of a globally unassembled vector.

   Collective on Vec

   Input Parameter:
.  v - the vector

   Output Parameter:
.  sum - the result

   Level: beginner

   Concepts: sum^of vector entries

.seealso: VecNorm()
@*/
PetscErrorCode  VecUASum(Vec v,PetscScalar *sum)
{
  PetscErrorCode    ierr;
  PetscInt          i,n;
  Vec               vi;
  PetscScalar       lsum = 0.0;
  const PetscScalar *x,*m;
  Vec_UNASM         *xi;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidScalarPointer(sum,2);
  xi   = (Vec_UNASM*)v->data;
  if (!xi->multiplicity) SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_SUP,"You should first call VecUnAsmSetMultiplicity");
  vi   = xi->vlocal;
  ierr = VecGetLocalSize(vi,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vi,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xi->multiplicity,&m);CHKERRQ(ierr);
  for (i=0; i<n; i++) lsum += x[i]/m[i];
  ierr = MPIU_Allreduce(&lsum,sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)v));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xi->multiplicity,&m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
