#include <private/einsvecimpl.h>
#include <../src/vec/einsvecunasm.h>

PetscClassId      VEC_EXCHANGE_CLASSID;

#undef __FUNCT__
#define __FUNCT__ "VecExchangeCreate"
/*@ VecExchangeCreate - Creates a VecExchange context. VecExchange is
     an object that is used to exchange data between neighboring
     processors in globally unassebled vectors.

   Input Parameter:
.  xin          - Globally unassembled vector.
.  n_neigh      - Number of neighbors. It is suppoused that I count myself.
.  neigh        - Array with neighbors. I consider myself a neighbor. I am the firt to be listed.
.  n_shared     - Array where n_shared[i] is the number of shared entries in xin with neighbor neigh[i].
.  shared       - Array where shared[i][j] is the j-th shared entry in xin in local numbering with neighbor neigh[i].
.  copy_mode    - The PetscCopyMode = {PETSC_COPY_VALUES,PETSC_OWN_POINTER,PETSC_USE_POINTER}. See PetscCopyMode.

   Output Parameter:
.  vec_exchange - the created VecExchange context

   Notes: others vectors different from xin can make use of the
   created VecExchange if they share the same neighboring information
   and layout. Despite the fact that the current processor is
   cosidered as another neighbor, it is not used in the
   implementation. This is inhereted from the implementation of the
   ISLocalToGlobalMappingGetInfo().

  Level: beginner


.seealso: VecExchangeDestroy(), VecExchangeBegin(), VecExchangeEnd()
@*/
PETSC_EXTERN PetscErrorCode VecExchangeCreate(Vec xin,PetscInt n_neigh,PetscInt* neigh,PetscInt* n_shared,PetscInt** shared,
					      PetscCopyMode copy_mode,VecExchange* vec_exchange)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  VecExchange    ctx;
  PetscInt       i,j,rank;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(xin,VEC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)xin,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Cannot use VecExchange with a non globally unassembled vector");
  PetscValidPointer(neigh,3);
  PetscValidPointer(n_shared,4);
  PetscValidPointer(shared,5);
  
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = PetscHeaderCreate(ctx,VEC_EXCHANGE_CLASSID,"VecExchange","VecExchange","Vec",comm,VecExchangeDestroy,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  ctx->copy_mode = copy_mode;
  ctx->n_neigh   = n_neigh-1;
  j = 0;
  for (i=1;i<n_neigh;i++) j += n_shared[i];
  if (copy_mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc3(ctx->n_neigh,&ctx->neigh,ctx->n_neigh,&ctx->n_shared,ctx->n_neigh,&ctx->shared);CHKERRQ(ierr);
    ierr = PetscMalloc1(j,&ctx->shared[0]);CHKERRQ(ierr);
    ierr = PetscMemcpy(ctx->neigh,neigh+1,ctx->n_neigh*sizeof(PetscInt));CHKERRQ(ierr);    
    ierr = PetscMemcpy(ctx->n_shared,n_shared+1,ctx->n_neigh*sizeof(PetscInt));CHKERRQ(ierr);    
    for (i=1;i<ctx->n_neigh;i++) ctx->shared[i] = ctx->shared[i-1]+ctx->n_shared[i-1];
    for (i=0;i<ctx->n_neigh;i++) {
      ierr = PetscMemcpy(ctx->shared[i],shared[i+1],n_shared[i+1]*sizeof(PetscInt));CHKERRQ(ierr);
    }

  } else if (copy_mode == PETSC_USE_POINTER){
    ctx->neigh     = neigh+1;
    ctx->n_shared  = n_shared+1;
    ierr = PetscMalloc1(ctx->n_neigh,&ctx->shared);CHKERRQ(ierr);   
    for (i=0;i<ctx->n_neigh;i++) ctx->shared[i] = shared[i+1];
  }else SETERRQ(comm,PETSC_ERR_SUP,"Cannot currently use PETSC_OWN_POINTER");
  
  ierr = PetscMalloc1(ctx->n_neigh,&ctx->r_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->n_neigh,&ctx->s_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->n_neigh,&ctx->work_vecs);CHKERRQ(ierr); 
  ierr = PetscMalloc1(j,&ctx->work_vecs[0]);CHKERRQ(ierr);
  for (i=1;i<ctx->n_neigh;i++) ctx->work_vecs[i] = ctx->work_vecs[i-1]+ctx->n_shared[i-1];
  
  ctx->ops->destroy  = 0;
  ctx->ops->view     = 0;
  ctx->ops->end      = 0;
  ctx->ops->begin    = 0;
  
  *vec_exchange = ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecExchangeDestroy"
/*@ VecExchangeDestroy - Destroys a VecExchange context.

  Input Parameter:
.  vec_exchange - the VecExchange context

  Level: beginner

.seealso: VecExchangeCreate(), VecExchangeBegin(), VecExchangeEnd()
@*/
PETSC_EXTERN PetscErrorCode VecExchangeDestroy(VecExchange* vec_exchange)
{
  PetscErrorCode ierr;
  VecExchange    ctx;
  
  PetscFunctionBegin;
  PetscValidPointer(vec_exchange,1);
  ctx = *vec_exchange; *vec_exchange = NULL;
  if (!ctx) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(ctx,VEC_EXCHANGE_CLASSID,1);
  if (--((PetscObject)ctx)->refct > 0) PetscFunctionReturn(0);
  if (ctx->ops->destroy) {ierr = (*ctx->ops->destroy)(ctx);CHKERRQ(ierr);}
  
  if (ctx->copy_mode == PETSC_COPY_VALUES) {
    ierr = PetscFree3(ctx->neigh,ctx->n_shared,ctx->shared[0]);CHKERRQ(ierr);
    ierr = PetscFree(ctx->shared);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(ctx->shared);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->s_reqs);CHKERRQ(ierr);
  ierr = PetscFree(ctx->r_reqs);CHKERRQ(ierr);
  ierr = PetscFree(ctx->work_vecs[0]);CHKERRQ(ierr);
  ierr = PetscFree(ctx->work_vecs);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(&ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecExchangeBegin"
/*@ VecExchangeBegin - Begins the exchange of data in the given globally unassembled vector.

  Input Parameter:
.  ve           - the VecExchange context.
.  xin          - the globally unassembled vector that will exchange local components.
.  imode        - the insert mode. The supported values are INSERT_VALUES and ADD_VALUES.

  Level: beginner

.seealso: VecExchangeCreate(), VecExchangeEnd()
@*/
PETSC_EXTERN PetscErrorCode VecExchangeBegin(VecExchange ve,Vec xin,InsertMode imode)
{
  PetscErrorCode      ierr;
  PetscInt            i;
  IS                  isindex;
  Vec                 vec;
  const PetscScalar   *array_s;
  PetscMPIInt         i_mpi;
  MPI_Comm            comm;
  Vec_UNASM           *xi;
  PetscBool           flg;
  PetscMPIInt         tag;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ve,VEC_EXCHANGE_CLASSID,1);
  PetscValidHeaderSpecific(xin,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)xin,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Cannot use VecExchange with a non globally unassembled vector");

  if (ve->ops->begin) {
    ierr = (*ve->ops->begin)(ve,xin,imode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  xi   = (Vec_UNASM*)xin->data;
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(comm, &tag);CHKERRQ(ierr);
  for (i=0; i<ve->n_neigh; i++){
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ve->n_shared[i],ve->shared[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
    ierr = VecGetSubVector(xi->vlocal,isindex,&vec);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = PetscMPIIntCast(ve->neigh[i],&i_mpi);CHKERRQ(ierr);   
    ierr = MPI_Isend(array_s,ve->n_shared[i],MPIU_SCALAR,i_mpi,tag,comm,&ve->s_reqs[i]);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec,&array_s);CHKERRQ(ierr);   
    ierr = VecRestoreSubVector(xi->vlocal,isindex,&vec);CHKERRQ(ierr);
    ierr = ISDestroy(&isindex);CHKERRQ(ierr);
  }

  for (i=0; i<ve->n_neigh; i++){
    ierr = PetscMPIIntCast(ve->neigh[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Irecv(ve->work_vecs[i],ve->n_shared[i],MPIU_SCALAR,i_mpi,tag,comm,&ve->r_reqs[i]);CHKERRQ(ierr);    
  }
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecExchangeEnd"
/*@ VecExchangeEnd - Begins the exchange of data in the given globally unassembled vector.

  Input Parameter:
.  ve           - the VecExchange context.
.  xin          - the globally unassembled vector that will exchange local components.
.  imode        - the insert mode. The supported values are INSERT_VALUES and ADD_VALUES.

  Level: beginner

.seealso: VecExchangeCreate(), VecExchangeBegin()
@*/
PETSC_EXTERN PetscErrorCode VecExchangeEnd(VecExchange ve,Vec xin,InsertMode imode)
{
  PetscErrorCode ierr;
  PetscInt       i;
  MPI_Comm       comm;
  Vec_UNASM      *xi;
  PetscBool      flg;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ve,VEC_EXCHANGE_CLASSID,1);
  PetscValidHeaderSpecific(xin,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)xin,VECMPIUNASM,&flg);CHKERRQ(ierr);
  if(!flg) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Cannot use VecExchange with a non globally unassembled vector");

  if (ve->ops->end) {
    ierr = (*ve->ops->end)(ve,xin,imode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  xi   = (Vec_UNASM*)xin->data;
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr); 
  ierr = MPI_Waitall(ve->n_neigh,ve->r_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(ve->n_neigh,ve->s_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

  for (i=0; i<ve->n_neigh; i++){   
    ierr = VecSetValues(xi->vlocal,ve->n_shared[i],ve->shared[i],ve->work_vecs[i],imode);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}


