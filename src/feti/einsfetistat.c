#include <../src/feti/einsfetistat.h>
#include <einsksp.h>
#include <einssys.h>
 

#undef __FUNCT__
#define __FUNCT__ "FETIDestroy_FETISTAT"
static PetscErrorCode FETIDestroy_FETISTAT(FETI ft)
{
  PetscErrorCode ierr;
  FETI_1         *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  if (!ft1) PetscFunctionReturn(0);
  ierr = VecDestroy(&ft1->local_e);CHKERRQ(ierr);
  ierr = VecDestroy(&ft1->alpha_local);CHKERRQ(ierr);
  ierr = MatDestroy(&ft1->rbm);CHKERRQ(ierr);
  ierr = PetscFree(ft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETISTAT"
static PetscErrorCode FETISetUp_FETISTAT(FETI ft)
{
  PetscErrorCode    ierr;   
  Subdomain         sd = ft->subdomain;
  PetscObjectState  mat_state;
  FETI_1            *ft1 = (FETI_1*)ft->data;
  
  PetscFunctionBegin;
  if (ft->state==FETI_STATE_INITIAL) {
    ierr = FETIBuildLambdaAndB(ft);CHKERRQ(ierr);
    ierr = FETIScalingSetUp(ft);CHKERRQ(ierr);
    ierr = FETISetUpNeumannSolverAndPerformFactorization(ft,PETSC_TRUE);CHKERRQ(ierr);
    ierr = FETIBuildInterfaceProblem(ft);CHKERRQ(ierr);
    ierr = FETIBuildInterfaceKSP(ft);CHKERRQ(ierr);

    ierr = FETICSSetUp(ft->ftcs);CHKERRQ(ierr);
    ierr = FETICSComputeCoarseBasis(ft->ftcs,&ft->localG,&ft1->rbm);CHKERRQ(ierr);

    /* compute matrix local_e */
    if(ft->n_cs){
      ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&ft1->local_e);CHKERRQ(ierr);
      ierr = MatMultTranspose(ft1->rbm,sd->localRHS,ft1->local_e);CHKERRQ(ierr);
    }
    ierr = FETISetInterfaceProblemRHS(ft);CHKERRQ(ierr);

    /* setup projection */
    ierr = FETIPJSetUp(ft->ftpj);CHKERRQ(ierr);
    ierr = FETIPJFactorizeCoarseProblem(ft->ftpj);CHKERRQ(ierr);
    /* creates alpha_local vector for holding local coefficients for vector with rigid body modes */
    if (ft->n_cs){ ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&ft1->alpha_local);CHKERRQ(ierr); }
  } else {
    ierr = PetscObjectStateGet((PetscObject)sd->localA,&mat_state);CHKERRQ(ierr);
    if (mat_state>ft->mat_state) {
      ierr = PetscObjectStateSet((PetscObject)ft->F,mat_state);CHKERRQ(ierr);  
      if (ft->resetup_pc_interface) {
	PC pc;
	ierr = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
	ierr = PCSetUp(pc);CHKERRQ(ierr);
      }
      ierr = FETISetUpNeumannSolverAndPerformFactorization(ft,PETSC_TRUE);CHKERRQ(ierr);
      ft->mat_state = mat_state;
    }
    /* compute matrix local_e */
    if(ft->n_cs){ ierr = MatMultTranspose(ft1->rbm,sd->localRHS,ft1->local_e);CHKERRQ(ierr);}
    ierr = FETISetInterfaceProblemRHS(ft);CHKERRQ(ierr);
  }

  ierr = FETIPJComputeInitialCondition(ft->ftpj,ft1->local_e,ft->lambda_global);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISolve_FETISTAT"
static PetscErrorCode FETISolve_FETISTAT(FETI ft, Vec u){
  PetscErrorCode    ierr;
  FETI_1            *ft1 = (FETI_1*)ft->data;
  Subdomain         sd = ft->subdomain;
  Vec               lambda_local;
  
  PetscFunctionBegin;
  /* Solve interface problem */
  ierr = KSPSolve(ft->ksp_interface,ft->d,ft->lambda_global);CHKERRQ(ierr);
  /* Get residual of the interface problem */
  ierr = KSPGetResidual(ft->ksp_interface,&ft1->res_interface);CHKERRQ(ierr);
  /* compute alpha_local */
  if(ft->n_cs) { ierr = FETIPJComputeAlphaNullSpace(ft->ftpj,ft1->res_interface,ft1->alpha_local);CHKERRQ(ierr);}
  /* computing B_delta^T*lambda */
  ierr = VecUnAsmGetLocalVectorRead(ft->lambda_global,&lambda_local);CHKERRQ(ierr);
  ierr = MatMultTranspose(ft->B_delta,lambda_local,sd->vec1_B);CHKERRQ(ierr);
  ierr = VecSet(sd->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,sd->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(ft->lambda_global,lambda_local);CHKERRQ(ierr);
  /* computing f - B_delta^T*lambda */
  ierr = VecAYPX(sd->vec1_N,-1.0,sd->localRHS);CHKERRQ(ierr);   
  /* Application of the already factorized pseudo-inverse */
  ierr = MatMumpsSetIcntl(ft->F_neumann,25,0);CHKERRQ(ierr);
  ierr = MatSolve(ft->F_neumann,sd->vec1_N,u);CHKERRQ(ierr);
  if (ft->n_cs) {
    /* computing R*alpha */
    ierr = MatMult(ft1->rbm,ft1->alpha_local,sd->vec1_N);CHKERRQ(ierr);
    /* computing u = A^+*(f - B_delta^T*lambda) + R*alpha */
    ierr = VecAXPY(u,-1.0,sd->vec1_N);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISTATSetDefaultOptions"
/*@
   FETISTATSetDefaultOptions - Sets default options for the FETISTAT
   solver. Mainly, it sets every KSP to MUMPS and sets fully redudant
   lagrange multipliers.

   Input: Input taken by PetscOptionsInsert()
.  argc   -  number of command line arguments
.  args   -  the command line arguments
.  file   -  optional file

   Level: beginner

.keywords: FETISTAT

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode FETISTATSetDefaultOptions(int *argc,char ***args,const char file[])
{
  PetscErrorCode    ierr;
  char mumps_options[]        = "-feti_pc_dirichlet_pc_factor_mat_solver_package mumps \
                                 -feti_pc_dirichlet_mat_mumps_icntl_7 2                \
                                 -feti_pj1level_pc_coarse_pc_factor_mat_solver_package mumps   \
                                 -feti_pj1level_pc_coarse_mat_mumps_icntl_7 2";
  char other_options[]        = "-feti_fullyredundant             \
                                 -feti_scaling_type scmultiplicity \
                                 -feti_pj1level_destroy_coarse";
  
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscOptionsInsertString(NULL,mumps_options);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsInsertString(NULL,other_options);CHKERRQ(ierr);
  ierr = PetscOptionsInsert(NULL,argc,args,file);CHKERRQ(ierr);
    
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "FETICreate_FETISTAT"
/*@
   FETISTAT - Implementation of the FETI method for static problems. Some comments about options can be put here!

   Options database:
.  -feti_fullyredundant: use fully redundant Lagrange multipliers.
.  -feti_interface_<ksp or pc option>: options for the KSP for the interface problem
.  -feti_neumann_<ksp or pc option>: for setting pc and ksp options for the neumann solver. 
.  -feti_pc_dirichilet_<ksp or pc option>: options for the KSP or PC to use for solving the Dirichlet problem
   associated to the Dirichlet preconditioner
.  -feti_scaling_type - Sets the scaling type
.  -feti_scaling_factor - Sets a scaling factor different from one
.  -feti_pj1level_destroy_coarse - If set, the matrix of the coarse problem, that is (G^T*G) or (G^T*Q*G), will be destroyed after factorization.
.  -feti_pj1level_pc_coarse_<ksp or pc option>: options for the KSP for the coarse problem

   Level: beginner

.keywords: FETI, FETISTAT
@*/
PetscErrorCode FETICreate_FETISTAT(FETI ft);
PetscErrorCode FETICreate_FETISTAT(FETI ft)
{
  PetscErrorCode      ierr;
  FETI_1*             feti1;

  PetscFunctionBegin;
  ierr      = PetscNewLog(ft,&feti1);CHKERRQ(ierr);
  ft->data  = (void*)feti1;

  feti1->res_interface         = 0;
  feti1->alpha_local           = 0;
  feti1->rbm                   = 0;
  ft->ftcs_type                = CS_RIGID_BODY_MODES;
  /* function pointers */
  ft->ops->setup               = FETISetUp_FETISTAT;
  ft->ops->destroy             = FETIDestroy_FETISTAT;
  ft->ops->setfromoptions      = 0;
  ft->ops->computesolution     = FETISolve_FETISTAT;
  ft->ops->view                = 0;

  ft->ftpj_type = PJ_FIRST_LEVEL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
