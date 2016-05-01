#include <../src/ksp/einspjcgimpl.h>
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt*);

static PetscErrorCode KSPGetProjection_PJCG(KSP,KSP_PROJECTION**);
static PetscErrorCode KSPSetUp_PJCG(KSP);
static PetscErrorCode KSPSolve_PJCG(KSP);
static PetscErrorCode KSPDestroy_PJCG(KSP);
static PetscErrorCode KSPView_PJCG(KSP,PetscViewer);
static PetscErrorCode KSPSetFromOptions_PJCG(PetscOptionItems*,KSP);
static PetscErrorCode KSPGetResidual_PJCG(KSP,Vec*);

const char *const KSPPJCGTruncationTypes[]     = {"STANDARD","NOTAY","KSPPJCGTrunctionTypes","KSP_PJCG_TRUNC_TYPE_",0};

#define KSPPJCG_DEFAULT_MMAX 30          /* maximum number of search directions to keep */
#define KSPPJCG_DEFAULT_NPREALLOC 10     /* number of search directions to preallocate */
#define KSPPJCG_DEFAULT_VECB 5           /* number of search directions to allocate each time new direction vectors are needed */
#define KSPPJCG_DEFAULT_TRUNCSTRAT KSP_PJCG_TRUNC_TYPE_NOTAY

#undef __FUNCT__
#define __FUNCT__ "KSPGetProjection_PJCG"
static PetscErrorCode KSPGetProjection_PJCG(KSP ksp,KSP_PROJECTION **pj)
{
  KSP_PJCG       *cg = (KSP_PJCG*)ksp->data; 
  PetscFunctionBegin;
  *pj = &cg->pj;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPGetResidual_PJCG"
static PetscErrorCode KSPGetResidual_PJCG(KSP ksp,Vec *res)
{
  PetscFunctionBegin;
  if (!ksp->work) SETERRQ(PetscObjectComm((PetscObject)ksp),1,"The residual vector is not yet allocated (=NULL).");
  *res = ksp->work[0];
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPAllocateVectors_PJCG"
static PetscErrorCode KSPAllocateVectors_PJCG(KSP ksp, PetscInt nvecsneeded, PetscInt chunksize)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  KSP_PJCG        *cg = (KSP_PJCG*)ksp->data;
  PetscInt        nnewvecs, nvecsprev;

  PetscFunctionBegin;
  /* Allocate enough new vectors to add chunksize new vectors, reach nvecsneedtotal, or to reach mmax+1, whichever is smallest */
  if (cg->nvecs < PetscMin(cg->mmax+1,nvecsneeded)){
    nvecsprev = cg->nvecs;
    nnewvecs = PetscMin(PetscMax(nvecsneeded-cg->nvecs,chunksize),cg->mmax+1-cg->nvecs);
    ierr = KSPCreateVecs(ksp,nnewvecs,&cg->pCvecs[cg->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,cg->pCvecs[cg->nchunks]);CHKERRQ(ierr);
    ierr = KSPCreateVecs(ksp,nnewvecs,&cg->pPvecs[cg->nchunks],0,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParents((PetscObject)ksp,nnewvecs,cg->pPvecs[cg->nchunks]);CHKERRQ(ierr);
    cg->nvecs += nnewvecs;
    for (i=0;i<nnewvecs;++i){
      cg->Cvecs[nvecsprev + i] = cg->pCvecs[cg->nchunks][i];
      cg->Pvecs[nvecsprev + i] = cg->pPvecs[cg->nchunks][i];
    }
    cg->chunksizes[cg->nchunks] = nnewvecs;
    ++cg->nchunks;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_PJCG"
static PetscErrorCode KSPSetUp_PJCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PJCG        *cg = (KSP_PJCG*)ksp->data;
  PetscInt       maxit = ksp->max_it;
  const PetscInt nworkstd = 4;

  PetscFunctionBegin;

  /* Allocate "standard" work vectors (not including the basis and transformed basis vectors) */
  ierr = KSPSetWorkVecs(ksp,nworkstd);CHKERRQ(ierr);

  /* Allocated space for pointers to additional work vectors
   note that mmax is the number of previous directions, so we add 1 for the current direction,
   and an extra 1 for the prealloc (which might be empty) */
  ierr = PetscMalloc5(cg->mmax+1,&cg->Pvecs,cg->mmax+1,&cg->Cvecs,cg->mmax+1,&cg->pPvecs,cg->mmax+1,&cg->pCvecs,cg->mmax+2,&cg->chunksizes);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ksp,(PetscLogDouble)(2*(cg->mmax+1)*sizeof(Vec*) + 2*(cg->mmax + 1)*sizeof(Vec**) + (cg->mmax + 2)*sizeof(PetscInt)));CHKERRQ(ierr);

  /* Preallocate additional work vectors */
  ierr = KSPAllocateVectors_PJCG(ksp,cg->nprealloc,cg->nprealloc);CHKERRQ(ierr);
  /*
  If user requested computations of eigenvalues then allocate work
  work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc4(maxit,&cg->e,maxit,&cg->d,maxit,&cg->ee,maxit,&cg->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,(PetscLogDouble)(2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal))));CHKERRQ(ierr);

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_PJCG"
static PetscErrorCode KSPSolve_PJCG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,k,idx,mi;
  KSP_PJCG        *cg = (KSP_PJCG*)ksp->data;
  PetscScalar    alpha=0.0,beta = 0.0,dpi;
  PetscReal      dp=0.0;
  Vec            B,R,W,Z,Zp,X,Pcurr,Ccurr;
  Mat            Amat,Pmat;
  PetscInt       eigs = ksp->calc_sings; /* Variables for eigen estimation - START*/
  PetscInt       stored_max_it = ksp->max_it;
  PetscScalar    alphaold = 0,betaold = 1.0,*e = 0,*d = 0;/* Variables for eigen estimation  - FINISH */
  KSP_PROJECTION *pj = &cg->pj;
  
  PetscFunctionBegin;

#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))
#define VecXMDot(a,b,c,d) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecMDot(a,b,c,d) : VecMTDot(a,b,c,d))

  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  W             = ksp->work[2];
  Zp            = ksp->work[3];
  
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);
  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  /* Compute initial residual needed for convergence check*/
  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);                    /*   r <- b - Ax     */
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*   r <- b (x is 0) */
  }
  if(pj->project) {ierr = (*pj->project)(pj->ctxProj,R,W);CHKERRQ(ierr);} /* w = P^T*r */
  ierr = VecNorm(W,NORM_2,&dp);CHKERRQ(ierr);

  /* Initial Convergence Check */
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;
  ierr       = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Apply PC if not already done for convergence check */
  ierr = KSP_PCApply(ksp,W,Zp);CHKERRQ(ierr);               /*   z <- Bw         */
  if(pj->reproject) {ierr = (*pj->reproject)(pj->ctxReProj,Zp,Z);CHKERRQ(ierr);}

  i = 0;
  do {
    ksp->its = i+1;

    /*  If needbe, allocate a new chunk of vectors in P and C */
    ierr = KSPAllocateVectors_PJCG(ksp,i+1,cg->vecb);CHKERRQ(ierr);

    /* Note that we wrap around and start clobbering old vectors */
    idx = i % (cg->mmax+1);
    Pcurr = cg->Pvecs[idx];
    Ccurr = cg->Cvecs[idx];

    /* Compute a new column of P (Currently does not support modified G-S or iterative refinement)*/
    switch(cg->truncstrat){
      case KSP_PJCG_TRUNC_TYPE_NOTAY :
        mi = PetscMax(1,i%(cg->mmax+1));
        break;
      case KSP_PJCG_TRUNC_TYPE_STANDARD :
        mi = cg->mmax;
        break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unrecognized PJCG Truncation Strategy");CHKERRQ(ierr);
    }
    ierr = VecCopy(Z,Pcurr);CHKERRQ(ierr);

    {
      PetscInt l,ndots;

      l = PetscMax(0,i-mi);
      ndots = i-l;
      if (ndots){
        PetscInt    j;
        Vec         *Pold,  *Cold;
        PetscScalar *dots;

        ierr = PetscMalloc3(ndots,&dots,ndots,&Cold,ndots,&Pold);CHKERRQ(ierr);
        for(k=l,j=0;j<ndots;++k,++j){
          idx = k % (cg->mmax+1);
          Cold[j] = cg->Cvecs[idx];
          Pold[j] = cg->Pvecs[idx];
        }
        ierr = VecXMDot(Z,ndots,Cold,dots);CHKERRQ(ierr);
        for(k=0;k<ndots;++k){
          dots[k] = -dots[k];
        }
        ierr = VecMAXPY(Pcurr,ndots,dots,Pold);CHKERRQ(ierr);
        ierr = PetscFree3(dots,Cold,Pold);CHKERRQ(ierr);
      }
    }

    /* Update X and R */
    betaold = beta;
    ierr = VecXDot(Pcurr,W,&beta);CHKERRQ(ierr);                 /*  beta <- pi'*r       */
    ierr = KSP_MatMult(ksp,Amat,Pcurr,Ccurr);CHKERRQ(ierr);      /*  w <- A*pi (stored in ci)   */
    ierr = VecXDot(Pcurr,Ccurr,&dpi);CHKERRQ(ierr);              /*  dpi <- pi'*w        */
    alphaold = alpha;
    alpha = beta / dpi;                                          /*  alpha <- beta/dpi    */
    ierr = VecAXPY(X,alpha,Pcurr);CHKERRQ(ierr);                 /*  x <- x + alpha * pi  */
    ierr = VecAXPY(R,-alpha,Ccurr);CHKERRQ(ierr);                /*  r <- r - alpha * wi  */

    if(pj->project) {ierr = (*pj->project)(pj->ctxProj,R,W);CHKERRQ(ierr);}
    ierr = VecNorm(W,NORM_2,&dp);CHKERRQ(ierr);
   
    /* Check for convergence */
    ksp->rnorm = dp;
    KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    ierr = KSP_PCApply(ksp,W,Zp);CHKERRQ(ierr);               /*   z <- Br         */
    if(pj->reproject) {ierr = (*pj->reproject)(pj->ctxReProj,Zp,Z);CHKERRQ(ierr);}
    
    /* Compute current C (which is W/dpi) */
    ierr = VecScale(Ccurr,1.0/dpi);CHKERRQ(ierr);              /*   w <- ci/dpi   */

    /* --->>> Begin eigen values computation */
    if (eigs) {
      if (i > 0) {
        if (ksp->max_it != stored_max_it) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(beta/betaold))/alphaold;
        d[i] = PetscSqrtReal(PetscAbsScalar(beta/betaold))*e[i] + 1.0/alpha;
      } else {
        d[i] = PetscSqrtReal(PetscAbsScalar(beta))*e[i] + 1.0/alpha;
      }
    }
    /* ---<<< End eigen values computation */
    
    ++i;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  if (eigs) cg->ned = ksp->its-1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_PJCG"
static PetscErrorCode KSPDestroy_PJCG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  KSP_PJCG        *cg = (KSP_PJCG*)ksp->data;

  PetscFunctionBegin;

  /* Destroy "standard" work vecs */
  VecDestroyVecs(ksp->nwork,&ksp->work);

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGetProjecion_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGetResidual_C",NULL);CHKERRQ(ierr);
  
  /* Destroy P and C vectors and the arrays that manage pointers to them */
  if (cg->nvecs){
    for (i=0;i<cg->nchunks;++i){
      ierr = VecDestroyVecs(cg->chunksizes[i],&cg->pPvecs[i]);CHKERRQ(ierr);
      ierr = VecDestroyVecs(cg->chunksizes[i],&cg->pCvecs[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree5(cg->Pvecs,cg->Cvecs,cg->pPvecs,cg->pCvecs,cg->chunksizes);CHKERRQ(ierr);
  /* free space used for singular value calculations */
  if (ksp->calc_sings) {
    ierr = PetscFree4(cg->e,cg->d,cg->ee,cg->dd);CHKERRQ(ierr);
  }
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPView_PJCG"
static PetscErrorCode KSPView_PJCG(KSP ksp,PetscViewer viewer)
{
  KSP_PJCG        *cg = (KSP_PJCG*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii,isstring;
  const char     *truncstr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);

  if (cg->truncstrat == KSP_PJCG_TRUNC_TYPE_STANDARD) truncstr = "Using standard truncation strategy";
  else if (cg->truncstrat == KSP_PJCG_TRUNC_TYPE_NOTAY) truncstr = "Using Notay's truncation strategy";
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Undefined PJCG truncation strategy");

  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  PJCG: m_max=%D\n",cg->mmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  PJCG: preallocated %D directions\n",PetscMin(cg->nprealloc,cg->mmax+1));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  PJCG: %s\n",truncstr);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,"m_max %D nprealloc %D %s",cg->mmax,cg->nprealloc,truncstr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPPJCGSetMmax"
/*@
  KSPPJCGSetMmax - set the maximum number of previous directions PJCG will store for orthogonalization

  Note: mmax + 1 directions are stored (mmax previous ones along with a current one)
  and whether all are used in each iteration also depends on the truncation strategy
  (see KSPPJCGSetTruncationType())

  Logically Collective on KSP

  Input Parameters:
+  ksp - the Krylov space context
-  mmax - the maximum number of previous directions to orthogonalize againt

  Level: intermediate

  Options Database:
. -ksp_cg_mmax <N>

.seealso: KSPPJCG, KSPPJCGGetTruncationType(), KSPPJCGGetNprealloc()
@*/
PetscErrorCode KSPPJCGSetMmax(KSP ksp,PetscInt mmax)
{
  KSP_PJCG *cg = (KSP_PJCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,mmax,2);
  cg->mmax = mmax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPPJCGGetMmax"
/*@
  KSPPJCGGetMmax - get the maximum number of previous directions PJCG will store

  Note: PJCG stores mmax+1 directions at most (mmax previous ones, and one current one)

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  mmax - the maximum number of previous directons allowed for orthogonalization

  Options Database:
. -ksp_cg_mmax <N>

   Level: intermediate

.keywords: KSP, PJCG, truncation

.seealso: KSPPJCG, KSPPJCGGetTruncationType(), KSPPJCGGetNprealloc(), KSPPJCGSetMmax()
@*/

PetscErrorCode KSPPJCGGetMmax(KSP ksp,PetscInt *mmax)
{
  KSP_PJCG *cg=(KSP_PJCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *mmax = cg->mmax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPPJCGSetNprealloc"
/*@
  KSPPJCGSetNprealloc - set the number of directions to preallocate with PJCG

  Logically Collective on KSP

  Input Parameters:
+  ksp - the Krylov space context
-  nprealloc - the number of vectors to preallocate

  Level: advanced

  Options Database:
. -ksp_cg_nprealloc <N>

.seealso: KSPPJCG, KSPPJCGGetTruncationType(), KSPPJCGGetNprealloc()
@*/
PetscErrorCode KSPPJCGSetNprealloc(KSP ksp,PetscInt nprealloc)
{
  KSP_PJCG *cg=(KSP_PJCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,nprealloc,2);
  if(nprealloc > cg->mmax+1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot preallocate more than m_max+1 vectors");
  cg->nprealloc = nprealloc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPPJCGGetNprealloc"
/*@
  KSPPJCGGetNprealloc - get the number of directions preallocate by PJCG

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  nprealloc - the number of directions preallocated

  Options Database:
. -ksp_cg_nprealloc <N>

   Level: advanced

.keywords: KSP, PJCG, truncation

.seealso: KSPPJCG, KSPPJCGGetTruncationType(), KSPPJCGSetNprealloc()
@*/
PetscErrorCode KSPPJCGGetNprealloc(KSP ksp,PetscInt *nprealloc)
{
  KSP_PJCG *cg=(KSP_PJCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *nprealloc = cg->nprealloc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPPJCGSetTruncationType"
/*@
  KSPPJCGSetTruncationType - specify how many of its stored previous directions PJCG uses during orthoganalization

  Logically Collective on KSP

  KSP_PJCG_TRUNC_TYPE_STANDARD uses all (up to mmax) stored directions
  KSP_PJCG_TRUNC_TYPE_NOTAY uses the last max(1,mod(i,mmax)) stored directions at iteration i=0,1..

  Input Parameters:
+  ksp - the Krylov space context
-  truncstrat - the choice of strategy

  Level: intermediate

  Options Database:
. -ksp_cg_truncation_type <standard, notay>

  .seealso: KSPPJCGTruncationType, KSPPJCGGetTruncationType
@*/
PetscErrorCode KSPPJCGSetTruncationType(KSP ksp,KSPPJCGTruncationType truncstrat)
{
  KSP_PJCG *cg=(KSP_PJCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveEnum(ksp,truncstrat,2);
  cg->truncstrat=truncstrat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPPJCGGetTruncationType"
/*@
  KSPPJCGGetTruncationType - get the truncation strategy employed by PJCG

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.  truncstrat - the strategy type

  Options Database:
. -ksp_cg_truncation_type <standard, notay>

   Level: intermediate

.keywords: KSP, PJCG, truncation

.seealso: KSPPJCG, KSPPJCGSetTruncationType, KSPPJCGTruncationType
@*/
PetscErrorCode KSPPJCGGetTruncationType(KSP ksp,KSPPJCGTruncationType *truncstrat)
{
  KSP_PJCG *cg=(KSP_PJCG*)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  *truncstrat=cg->truncstrat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_PJCG"
static PetscErrorCode KSPSetFromOptions_PJCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PJCG        *cg=(KSP_PJCG*)ksp->data;
  PetscInt       mmax,nprealloc;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP PJCG Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_cg_mmax","Maximum number of search directions to store","KSPPJCGSetMmax",cg->mmax,&mmax,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPPJCGSetMmax(ksp,mmax);CHKERRQ(ierr); 
  }
  ierr = PetscOptionsInt("-ksp_cg_nprealloc","Number of directions to preallocate","KSPPJCGSetNprealloc",cg->nprealloc,&nprealloc,&flg);CHKERRQ(ierr);
  if (flg) { 
    ierr = KSPPJCGSetNprealloc(ksp,nprealloc);CHKERRQ(ierr); 
  }
  ierr = PetscOptionsEnum("-ksp_cg_truncation_type","Truncation approach for directions","KSPPJCGSetTruncationType",KSPPJCGTruncationTypes,(PetscEnum)cg->truncstrat,(PetscEnum*)&cg->truncstrat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      KSPPJCG - Implements the Flexible Conjugate Gradient method (PJCG)

  Options Database Keys:
+   -ksp_cg_mmax <N>
.   -ksp_cg_nprealloc <N>
-   -ksp_cg_truncation_type <standard,notay>

    Contributed by Patrick Sanan

   Notes:
   Supports left preconditioning only.

   Level: beginner

  References:
    1) Notay, Y."Flexible Conjugate Gradients", SIAM J. Sci. Comput. 22:4, pp 1444-1460, 2000

    2) Axelsson, O. and Vassilevski, P. S. "A Black Box Generalized Conjugate Gradient Solver with Inner Iterations and Variable-Step Preconditioning",
    SIAM J. Matrix Anal. Appl. 12:4, pp 625-44, 1991

 .seealso : KSPGCR, KSPFGMRES, KSPCG, KSPPJCGSetMmax(), KSPPJCGGetMmax(), KSPPJCGSetNprealloc(), KSPPJCGGetNprealloc(), KSPPJCGSetTruncationType(), KSPPJCGGetTruncationType()

M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_PJCG"
PETSC_EXTERN PetscErrorCode KSPCreate_PJCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_PJCG        *cg;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ksp,&cg);CHKERRQ(ierr);
  ksp->data = (void*)cg;

  cg->pj.ctxProj                      = 0;
  cg->pj.ctxReProj                    = 0;
  cg->pj.project                      = 0;
  cg->pj.reproject                    = 0;

#if !defined(PETSC_USE_COMPLEX)
  cg->type       = KSP_CG_SYMMETRIC;
#else
  cg->type       = KSP_CG_HERMITIAN;
#endif
  cg->mmax       = KSPPJCG_DEFAULT_MMAX;
  cg->nprealloc  = KSPPJCG_DEFAULT_NPREALLOC;
  cg->nvecs      = 0;
  cg->vecb       = KSPPJCG_DEFAULT_VECB;
  cg->nchunks    = 0;
  cg->truncstrat = KSPPJCG_DEFAULT_TRUNCSTRAT;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGetProjecion_C",KSPGetProjection_PJCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPGetResidual_C",KSPGetResidual_PJCG);CHKERRQ(ierr);
  
  ksp->ops->setup          = KSPSetUp_PJCG;
  ksp->ops->solve          = KSPSolve_PJCG;
  ksp->ops->destroy        = KSPDestroy_PJCG;
  ksp->ops->view           = KSPView_PJCG;
  ksp->ops->setfromoptions = KSPSetFromOptions_PJCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}

