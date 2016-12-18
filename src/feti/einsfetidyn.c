#include <einsksp.h>
#include <einspc.h>
#include <einssys.h>
#include <private/einsfetiimpl.h>

#undef __FUNCT__
#define __FUNCT__ "FETISetUp_FETIDYN"
static PetscErrorCode FETISetUp_FETIDYN(FETI ft)
{
  PetscErrorCode    ierr;
  Subdomain         sd = ft->subdomain;
  PetscObjectState  mat_state;
  
  PetscFunctionBegin;
  if (ft->state==FETI_STATE_INITIAL) {
    ierr = FETIBuildLambdaAndB(ft);CHKERRQ(ierr);
    ierr = FETIScalingSetUp(ft);CHKERRQ(ierr);
    ierr = FETISetUpNeumannSolverAndPerformFactorization(ft,PETSC_FALSE);CHKERRQ(ierr);
    ierr = FETIBuildInterfaceProblem(ft);CHKERRQ(ierr);
    ierr = FETIBuildInterfaceKSP(ft);CHKERRQ(ierr); /* the PC for the interface problem is setup here */
    ierr = FETICSSetUp(ft->ftcs);CHKERRQ(ierr);
    ierr = FETICSComputeCoarseBasis(ft->ftcs,&ft->localG,NULL);CHKERRQ(ierr);
    ierr = FETISetInterfaceProblemRHS(ft);CHKERRQ(ierr);

    /* set projection */
    ierr = FETIPJSetUp(ft->ftpj);CHKERRQ(ierr);     
    ierr = FETIPJGatherNeighborsCoarseBasis(ft->ftpj);CHKERRQ(ierr);
    ierr = FETIPJAssembleCoarseProblem(ft->ftpj);CHKERRQ(ierr);
    ierr = FETIPJFactorizeCoarseProblem(ft->ftpj);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectStateGet((PetscObject)sd->localA,&mat_state);CHKERRQ(ierr);
    if (mat_state>ft->mat_state) {
      ierr = PetscObjectStateSet((PetscObject)ft->F,mat_state);CHKERRQ(ierr);  
      if (ft->resetup_pc_interface) {
	PC pc;
	ierr = KSPGetPC(ft->ksp_interface,&pc);CHKERRQ(ierr);
	ierr = PCSetUp(pc);CHKERRQ(ierr);
      }
      ierr = FETISetUpNeumannSolverAndPerformFactorization(ft,PETSC_FALSE);CHKERRQ(ierr);
      if (ft->resetup_pc_interface) {
	ierr = FETICSComputeCoarseBasis(ft->ftcs,&ft->localG,NULL);CHKERRQ(ierr);
	ierr = FETIPJGatherNeighborsCoarseBasis(ft->ftpj);CHKERRQ(ierr);
      }
      ierr = FETIPJAssembleCoarseProblem(ft->ftpj);CHKERRQ(ierr);
      ierr = FETIPJFactorizeCoarseProblem(ft->ftpj);CHKERRQ(ierr);
      ft->mat_state = mat_state;
    }
    ierr = FETISetInterfaceProblemRHS(ft);CHKERRQ(ierr);
  }

  ierr = FETIPJComputeInitialCondition(ft->ftpj,ft->d,ft->lambda_global);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETISolve_FETIDYN"
static PetscErrorCode FETISolve_FETIDYN(FETI ft, Vec u){
  PetscErrorCode    ierr;
  Subdomain         sd = ft->subdomain;
  Vec               lambda_local;
  
  PetscFunctionBegin;
  /* Solve interface problem */
  ierr = KSPSolve(ft->ksp_interface,ft->d,ft->lambda_global);CHKERRQ(ierr);
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
  ierr = MatSolve(ft->F_neumann,sd->vec1_N,u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIDYNSetDefaultOptions"
/*@
   FETIDYNSetDefaultOptions - Sets default options for the FETIDYN
   solver. Mainly, it sets every KSP to MUMPS and sets fully redudant
   lagrange multipliers.

   Input: Input taken by PetscOptionsInsert()
.  argc   -  number of command line arguments
.  args   -  the command line arguments
.  file   -  optional file

   Level: beginner

.keywords: FETIDYN

.seealso: PetscOptionsInsert()
@*/
PetscErrorCode FETIDYNSetDefaultOptions(int *argc,char ***args,const char file[])
{
  PetscErrorCode    ierr;
  char mumps_options[]        = "-feti_pc_dirichlet_pc_factor_mat_solver_package mumps \
                                 -feti_pc_dirichlet_mat_mumps_icntl_7 2                \
                                 -feti_neumann_pc_factor_mat_solver_package mumps     \
                                 -feti_neumann_mat_mumps_icntl_7 2                    \
                                 -feti_pj2level_pc_coarse_pc_factor_mat_solver_package mumps   \
                                 -feti_pj2level_pc_coarse_mat_mumps_icntl_7 2";
  char other_options[]        = "-feti_fullyredundant             \
                                 -feti_scaling_type scmultiplicity";
  
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
#define __FUNCT__ "FETICreate_FETIDYN"
/*@
   FETIDYN - Implementation of the FETI method for dynamic problems. Some comments about options can be put here!

   Options database:
.  -feti_fullyredundant: use fully redundant Lagrange multipliers.
.  -feti_interface_<ksp or pc option>: options for the KSP for the interface problem
.  -feti_neumann_<ksp or pc option>: for setting pc and ksp options for the neumann solver. 
.  -feti_pc_dirichilet_<ksp or pc option>: options for the KSP or PC to use for solving the Dirichlet problem
   associated to the Dirichlet preconditioner
.  -feti_scaling_type - Sets the scaling type
.  -feti_pj2level_pc_coarse_<ksp or pc option>: options for the KSP for the coarse problem
.  -fetics_geneo_<option>: options for FETIDYN using GENEO modes (e.g.:-fetics_geneo_eps_nev sets the number of eigenvalues)

   Level: beginner

.keywords: FETI, FETIDYN
@*/
PetscErrorCode FETICreate_FETIDYN(FETI ft);
PetscErrorCode FETICreate_FETIDYN(FETI ft)
{
  PetscFunctionBegin;  
  /* function pointers */
  ft->ops->setup               = FETISetUp_FETIDYN;
  ft->ops->destroy             = 0;
  ft->ops->setfromoptions      = 0;
  ft->ops->computesolution     = FETISolve_FETIDYN;
  ft->ops->view                = 0;

  ft->ftpj_type = PJ_SECOND_LEVEL;
  PetscFunctionReturn(0);
}
EXTERN_C_END
