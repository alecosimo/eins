#include <../src/feti/einsfeti2.h>
#include <../src/pc/einspcdirichlet.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einssys.h>
#include <einspc.h>


#if defined(HAVE_SLEPC)

static PetscErrorCode MatMultBg_FETI2_GENEO(Mat,Vec,Vec);
static PetscErrorCode MatDestroyBg_FETI2_GENEO(Mat);
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
#define __FUNCT__ "FETI2ComputeMatrixG_GENEO"
/*@
   FETI2ComputeMatrixG_GENEO - Computes GENEO modes.

   Input Parameters:
.  ft - the FETI context

@*/
PetscErrorCode FETI2ComputeMatrixG_GENEO(FETI ft)
{
  PetscErrorCode    ierr;
  FETI_2            *ft2 = (FETI_2*)ft->data;
  GENEO_C           *gn  = ft2->geneo;
  PetscInt          nconv,nev,i;
  Vec               vec;
  PetscScalar       *pointer_vec;
  Subdomain         sd = ft->subdomain;
  
  PetscFunctionBegin;
  if (!gn) SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: GENEO must be first created");
  /* update status of preconditioners */
  if (PetscNot(gn->flg)) { ierr = PCSetUp(gn->pc_dirichlet);CHKERRQ(ierr);}

  /* solve eingenvalue problem */
  ierr = EPSSetOperators(gn->eps,gn->Bg,NULL);CHKERRQ(ierr); 
  ierr = EPSSolve(gn->eps);CHKERRQ(ierr);

  ierr = EPSGetDimensions(gn->eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = EPSGetConverged(gn->eps,&nconv);CHKERRQ(ierr);
  if (nconv<nev) SETERRQ2(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: some of the GENEO modes did not converged: nev: %d, nconv: %d",nev,nconv);
  /* augement and apply precondtioner */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,ft->n_lambda_local,NULL,&vec);CHKERRQ(ierr);
  ierr = MatDenseGetArray(ft2->localG,&pointer_vec);CHKERRQ(ierr);
  for (i=0;i<nev;i++) {
    ierr = VecPlaceArray(vec,(const PetscScalar*)(pointer_vec+ft->n_lambda_local*i));CHKERRQ(ierr);
    ierr = EPSGetEigenpair(gn->eps,i,NULL,NULL,sd->vec1_B,NULL);CHKERRQ(ierr);
    ierr = MatMult(ft->B_delta,sd->vec1_B,gn->vec_lb1);CHKERRQ(ierr);
    ierr = PCApplyLocal(gn->pc,gn->vec_lb1,vec);CHKERRQ(ierr);
    ierr = VecResetArray(vec);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(ft2->localG,&pointer_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatDestroyBg_FETI2_GENEO"
static PetscErrorCode MatDestroyBg_FETI2_GENEO(Mat A)
{
  GENEOMat_ctx    mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETICreate_FETI2_GENEO"
/*@
   FETICreate_FETI2_GENEO - Creates structures need for FETI2 GENEO.

   Input Parameters:
.  ft - the FETI context

@*/
PetscErrorCode FETICreate_FETI2_GENEO(FETI ft)
{
  PetscErrorCode  ierr;
  FETI_2          *ft2 = (FETI_2*)ft->data;
  PC              pc;
  PCFT_DIRICHLET  *pcd; 
  PCType          pctype;
  PetscBool       flg;
  PetscInt        n,nev;
  GENEO_C         *gn;
  GENEOMat_ctx    matctx;
  MPI_Comm        comm;
  
  PetscFunctionBegin;
  ierr       = PetscNewLog(ft,&gn);CHKERRQ(ierr);
  ft2->geneo = gn;

  ierr   = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr   = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
  gn->pc = pc;
  ierr   = PetscObjectReference((PetscObject) pc);CHKERRQ(ierr); 
  ierr   = PCGetType(pc,&pctype);CHKERRQ(ierr);
  ierr   = PetscStrcmp(PCFETI_DIRICHLET,pctype,&flg);CHKERRQ(ierr);
  if (flg) {
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
  ierr = MatCreateShell(PETSC_COMM_SELF,n,n,n,n,matctx,&gn->Bg);CHKERRQ(ierr);
  ierr = MatShellSetOperation(gn->Bg,MATOP_MULT,(void(*)(void))MatMultBg_FETI2_GENEO);CHKERRQ(ierr);
  ierr = MatShellSetOperation(gn->Bg,MATOP_DESTROY,(void(*)(void))MatDestroyBg_FETI2_GENEO);CHKERRQ(ierr);
  ierr = MatSetUp(gn->Bg);CHKERRQ(ierr);
  
  /* create and setup SLEPc solver */
  ierr = EPSCreate(PETSC_COMM_SELF,&gn->eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(gn->eps,gn->Bg,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(gn->eps,EPS_HEP);CHKERRQ(ierr); /* hermitanian problem */
  ierr = EPSSetType(gn->eps,EPSKRYLOVSCHUR);CHKERRQ(ierr); 
  ierr = EPSSetWhichEigenpairs(gn->eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetDimensions(gn->eps,3,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr); /* -eps_nev <nev> - Sets the number of eigenvalues */
  ierr = EPSSetStoppingTestFunction(gn->eps,EPSStoppingGeneo_Private,(void*)&gn->pc,NULL);CHKERRQ(ierr);
  ierr = EPSSetOptionsPrefix(gn->eps,"feti2_geneo_");CHKERRQ(ierr);
  ierr = EPSSetFromOptions(gn->eps);CHKERRQ(ierr);

  /* crete localG matrix */
  ierr = EPSGetDimensions(gn->eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ft->n_cs = nev;
  ierr = MatDestroy(&ft2->localG);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,ft->n_cs,NULL,&ft2->localG);CHKERRQ(ierr);
  ierr = MatSetOption(ft2->localG,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

  /* create working vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_lambda_local,&gn->vec_lb1);CHKERRQ(ierr);
  ierr = VecDuplicate(gn->vec_lb1,&gn->vec_lb2);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETI2_GENEO"
/*@
   FETISetUp_FETI2_GENEO - Setups structures need for FETI2 GENEO.

   Input Parameters:
.  ft - the FETI context

@*/
PetscErrorCode FETISetUp_FETI2_GENEO(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  PC             pc;
  PCType         pctype;
  GENEO_C        *gn = ft2->geneo;
  
  PetscFunctionBegin;
  if (!gn) SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: GENEO must be first created");
  ierr = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
  ierr = PetscStrcmp(PCFETI_DIRICHLET,pctype,&gn->flg);CHKERRQ(ierr);
  if (PetscNot(gn->flg)) { ierr = PCSetUp(gn->pc_dirichlet);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatMultBg_FETI2_GENEO"
static PetscErrorCode MatMultBg_FETI2_GENEO(Mat A,Vec x,Vec y)
{
  GENEOMat_ctx      mat_ctx;
  PetscErrorCode    ierr;
  FETI              ft;
  GENEO_C           *gn;
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
#define __FUNCT__ "FETIDestroy_FETI2_GENEO"
/*@
   FETIDestroy_FETI2_GENEO - Destroys structures generated by FETI2 with GENEO modes.

   Input Parameters:
.  ft - the FETI context

@*/
PetscErrorCode FETIDestroy_FETI2_GENEO(FETI ft)
{
  PetscErrorCode ierr;
  FETI_2         *ft2 = (FETI_2*)ft->data;
  GENEO_C        *gn  = ft2->geneo;
  
  PetscFunctionBegin;
  ierr = MatDestroy(&gn->Bg);CHKERRQ(ierr);
  ierr = PCDestroy(&gn->pc_dirichlet);CHKERRQ(ierr);
  ierr = PCDestroy(&gn->pc);CHKERRQ(ierr);
  ierr = VecDestroy(&gn->vec_lb1);CHKERRQ(ierr);
  ierr = VecDestroy(&gn->vec_lb2);CHKERRQ(ierr);
  ierr = EPSDestroy(&gn->eps);CHKERRQ(ierr);
  ierr = PetscFree(ft2->geneo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#endif
