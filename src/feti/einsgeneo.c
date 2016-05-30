#include <../src/feti/einsgeneo.h>
#include <../src/pc/einspcdirichlet.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einssys.h>
#include <einspc.h>


#if defined(HAVE_SLEPC)

static PetscErrorCode MatMultAg_GENEO(Mat,Vec,Vec);
static PetscErrorCode MatDestroyAg_GENEO(Mat);
static PetscErrorCode EPSStoppingGeneo_Private(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*);


#undef __FUNCT__
#define __FUNCT__ "EPSStoppingGeneo_Private"
static PetscErrorCode EPSStoppingGeneo_Private(EPS eps,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,EPSConvergedReason *reason,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       n2c;
  PC             pc;
  
  PetscFunctionBegin;
  ierr = EPSStoppingBasic(eps,its,max_it,nconv,nev,reason,ctx);CHKERRQ(ierr);
  if ((*reason == EPS_CONVERGED_TOL) || (*reason == EPS_DIVERGED_ITS)) {
    n2c = 1;
    pc = *((PC*)ctx);
    while (n2c) {
      ierr = PCApplyLocalWithPolling(pc,NULL,NULL,&n2c);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSComputeCoarseBasis_GENEO"
static PetscErrorCode FETICSComputeCoarseBasis_GENEO(FETICS ftcs,Mat *localG,PETSC_UNUSED Mat* R)
{
  PetscErrorCode    ierr;
  FETI              ft  = ftcs->feti;
  GENEO_CS          *gn = (GENEO_CS*)ftcs->data;
  PetscInt          nconv,nev,i;
  Vec               vec;
  PetscScalar       *pointer_vec;
  Subdomain         sd = ft->subdomain;
  
  PetscFunctionBegin;
  /* update status of preconditioners */
  if (PetscNot(gn->flg)) { ierr = PCSetUp(gn->pc_dirichlet);CHKERRQ(ierr);}

  /* solve eingenvalue problem */
  ierr = EPSSetOperators(gn->eps,gn->Ag,NULL);CHKERRQ(ierr); 
  ierr = EPSSolve(gn->eps);CHKERRQ(ierr);

  ierr = EPSGetDimensions(gn->eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = EPSGetConverged(gn->eps,&nconv);CHKERRQ(ierr);
  if (nconv<nev) SETERRQ2(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: some of the GENEO modes did not converged: nev: %d, nconv: %d",nev,nconv);
  /* augement and apply precondtioner */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,ft->n_lambda_local,NULL,&vec);CHKERRQ(ierr);
  ierr = MatDenseGetArray(gn->localG,&pointer_vec);CHKERRQ(ierr);
  for (i=0;i<nev;i++) {
    ierr = VecPlaceArray(vec,(const PetscScalar*)(pointer_vec+ft->n_lambda_local*i));CHKERRQ(ierr);
    ierr = EPSGetEigenpair(gn->eps,i,NULL,NULL,sd->vec1_B,NULL);CHKERRQ(ierr);
    ierr = MatMult(ft->B_delta,sd->vec1_B,gn->vec_lb1);CHKERRQ(ierr);
    ierr = PCApplyLocal(gn->pc,gn->vec_lb1,vec);CHKERRQ(ierr);
    ierr = VecResetArray(vec);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(gn->localG,&pointer_vec);CHKERRQ(ierr);
  
  ierr    = MatDestroy(localG);CHKERRQ(ierr);
  *localG = gn->localG;
  ierr    = PetscObjectReference((PetscObject)gn->localG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatDestroyAg_GENEO"
static PetscErrorCode MatDestroyAg_GENEO(Mat A)
{
  GENEOMat_ctx    mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSSetUp_GENEO"
static PetscErrorCode FETICSSetUp_GENEO(FETICS ftcs)
{
  PetscErrorCode   ierr;
  FETI             ft = ftcs->feti;
  PC               pc;
  PCType           pctype;
  GENEO_CS         *gn = (GENEO_CS*)ftcs->data;
  MPI_Comm         comm;
  PCFT_DIRICHLET   *pcd;
  PetscInt         n,nev;
  GENEOMat_ctx     matctx;
  
  PetscFunctionBegin;
  ierr   = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr   = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
  gn->pc = pc;
  ierr   = PetscObjectReference((PetscObject) pc);CHKERRQ(ierr); 
  ierr   = PCGetType(pc,&pctype);CHKERRQ(ierr);
  ierr   = PetscStrcmp(PCFETI_DIRICHLET,pctype,&gn->flg);CHKERRQ(ierr);
  if (gn->flg) {
    gn->pc_dirichlet = pc;
    ierr   = PetscObjectReference((PetscObject) pc);CHKERRQ(ierr); 
  } else {
    ierr = PCCreate(comm,&gn->pc_dirichlet);CHKERRQ(ierr);
    ierr = PCSetType(gn->pc_dirichlet,PCFETI_DIRICHLET);CHKERRQ(ierr);
    ierr = PCSetOperators(gn->pc_dirichlet,ft->F,ft->F);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)gn->pc_dirichlet,(PetscObject)ft,0);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)ft,(PetscObject)gn->pc_dirichlet);CHKERRQ(ierr);
    ierr = PCSetUp(gn->pc_dirichlet);CHKERRQ(ierr);
  }
  /* create MATSHELLs */
  /* creating the mat context for the MatShell for the eigenvalue problem */
  ierr = PetscNew(&matctx);CHKERRQ(ierr);
  matctx->ft = ft;
  matctx->gn = gn;
  pcd  = (PCFT_DIRICHLET*)(gn->pc_dirichlet->data);
  ierr = MatGetSize(pcd->Sj,&n,NULL);CHKERRQ(ierr);
  ierr = MatCreateShell(PETSC_COMM_SELF,n,n,n,n,matctx,&gn->Ag);CHKERRQ(ierr);
  ierr = MatShellSetOperation(gn->Ag,MATOP_MULT,(void(*)(void))MatMultAg_GENEO);CHKERRQ(ierr);
  ierr = MatShellSetOperation(gn->Ag,MATOP_DESTROY,(void(*)(void))MatDestroyAg_GENEO);CHKERRQ(ierr);
  ierr = MatSetUp(gn->Ag);CHKERRQ(ierr);
  
  /* create and setup SLEPc solver */
  ierr = EPSCreate(PETSC_COMM_SELF,&gn->eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(gn->eps,gn->Ag,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(gn->eps,EPS_HEP);CHKERRQ(ierr); /* hermitanian problem */
  ierr = EPSSetType(gn->eps,EPSKRYLOVSCHUR);CHKERRQ(ierr); 
  ierr = EPSSetWhichEigenpairs(gn->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetDimensions(gn->eps,3,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr); /* -eps_nev <nev> - Sets the number of eigenvalues */
  ierr = EPSSetStoppingTestFunction(gn->eps,EPSStoppingGeneo_Private,(void*)&gn->pc,NULL);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(gn->eps,"feti_geneo_");CHKERRQ(ierr);
  ierr = EPSSetFromOptions(gn->eps);CHKERRQ(ierr);

  /* crete localG matrix */
  ierr = EPSGetDimensions(gn->eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ft->n_cs = nev;
  PetscPrintf(PETSC_COMM_WORLD,"\n---------------------------------------- FETI2SetInterfaceProblemRHS_Private %p\n",gn->localG);
  ierr = MatDestroy(&gn->localG);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,ft->n_cs,NULL,&gn->localG);CHKERRQ(ierr);
  ierr = MatSetOption(gn->localG,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

  /* create working vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_lambda_local,&gn->vec_lb1);CHKERRQ(ierr);
  ierr = VecDuplicate(gn->vec_lb1,&gn->vec_lb2);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMultAg_GENEO"
static PetscErrorCode MatMultAg_GENEO(Mat A,Vec x,Vec y)
{
  GENEOMat_ctx      mat_ctx;
  PetscErrorCode    ierr;
  FETI              ft;
  GENEO_CS          *gn;
  Subdomain         sd;
  
  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,&mat_ctx);CHKERRQ(ierr);
  ft   = mat_ctx->ft;
  gn   = mat_ctx->gn;
  sd   = ft->subdomain;

  /* applying B^T*S_d*B */
  ierr = MatMult(ft->B_delta,x,gn->vec_lb1);CHKERRQ(ierr);
  ierr = PCApplyLocalWithPolling(gn->pc,gn->vec_lb1,gn->vec_lb2,NULL);CHKERRQ(ierr); 
  ierr = MatMultTranspose(ft->B_delta,gn->vec_lb2,sd->vec1_B);CHKERRQ(ierr);
  
  /* applying S^-1 */
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);  
  ierr = MatSolve(ft->F_neumann,sd->vec1_N,sd->vec2_N);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec2_N,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec2_N,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSDestroy_GENEO"
static PetscErrorCode FETICSDestroy_GENEO(FETICS ftcs)
{
  PetscErrorCode ierr;
  GENEO_CS       *gn = (GENEO_CS*)ftcs->data;
  
  PetscFunctionBegin;
  ierr = MatDestroy(&gn->Ag);CHKERRQ(ierr);
  ierr = MatDestroy(&gn->localG);CHKERRQ(ierr);
  ierr = PCDestroy(&gn->pc_dirichlet);CHKERRQ(ierr);
  ierr = PCDestroy(&gn->pc);CHKERRQ(ierr);
  ierr = VecDestroy(&gn->vec_lb1);CHKERRQ(ierr);
  ierr = VecDestroy(&gn->vec_lb2);CHKERRQ(ierr);
  ierr = EPSDestroy(&gn->eps);CHKERRQ(ierr);
  ierr = PetscFree(ftcs->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICSCreate_GENEO"
PETSC_EXTERN PetscErrorCode FETICSCreate_GENEO(FETICS);
PetscErrorCode FETICSCreate_GENEO(FETICS ftcs)
{
  PetscErrorCode  ierr;
  FETI            ft = ftcs->feti;
  GENEO_CS        *gn;
  
  PetscFunctionBegin;
  ierr       = PetscNewLog(ft,&gn);CHKERRQ(ierr);
  ftcs->data = (void*)gn;

  ierr  = PetscMemzero(gn,sizeof(GENEO_CS));CHKERRQ(ierr);

  ftcs->ops->setup               = FETICSSetUp_GENEO;
  ftcs->ops->destroy             = FETICSDestroy_GENEO;
  ftcs->ops->setfromoptions      = 0;
  ftcs->ops->computecoarsebasis  = FETICSComputeCoarseBasis_GENEO; 
  PetscFunctionReturn(0);
}


#endif
