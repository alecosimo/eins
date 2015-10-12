static char help[] = "Creates a sequential matrix and calls a direct solver to\n\
 solve a semi-definite system and computes the rigid body modes\n\n";

#include <petscksp.h>
#include <petsctime.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
#if !defined(PETSC_HAVE_MUMPS)
  SETERRQ(PETSC_COMM_WORLD,1,"This test requires MUMPS");
#endif

  KSP                ksp;
  PC                 pc;
  Mat                A,F;
  Mat                x,b;
  Vec                xv,bv;
  MPI_Comm           comm;
  PetscInt           i,n = 4,elems = 1,disp;
  PetscErrorCode     ierr;
  PetscScalar        L = 1.0, values[16] = {12,6*L,-12,6*L,6*L,4*L*L,-6*L,2*L*L,-12,-6*L,12,-6*L,6*L,2*L*L,-6*L,4*L*L};
  PetscBool          flg_mumps,flg_mumps_ch;
  PetscLogDouble     v1,v2,elapsed_time;
  
  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = MPI_COMM_SELF;
  
  /* n: number of rows of the matrix.*/
  ierr = PetscOptionsGetInt(NULL,"-elems",&elems,NULL);CHKERRQ(ierr);
  n = 2*(elems-1) + n;
		
  /* Construct the matrix.*/
  ierr = MatCreateSeqAIJ(comm,n,n,6,NULL,&A);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  for (i=0; i<elems; i++) {
    disp = 2*i;
    PetscInt idxRowCol[4] = {0+disp,1+disp,2+disp,3+disp};
    ierr = MatSetValuesBlocked(A,4,idxRowCol,4,idxRowCol,values,ADD_VALUES);CHKERRQ(ierr);
    //ierr = MatSetValues(A,4,idxRowCol,4,idxRowCol,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      
  /* Create linear solver context */
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);

  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  /* call MatGetFactor() to create F, a matrix object specifically suited for symbolic factorization */
  ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&F);CHKERRQ(ierr);

  /* sequential ordering */
  PetscInt  ival,icntl;
  PetscReal val;
  icntl = 7; ival = 2;
  ierr = MatMumpsSetIcntl(F,icntl,ival);CHKERRQ(ierr);

  /* threshhold for row pivot detection */
  ierr = MatMumpsSetIcntl(F,24,1);CHKERRQ(ierr);
  ierr = MatMumpsSetIcntl(F,21,0);CHKERRQ(ierr);
  icntl = 3; val = 1.e-6;
  ierr = MatMumpsSetCntl(F,icntl,val);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Get info from matrix factors and perform symbolic and numerical factorization
     (already checked!!!) */
  ierr = PetscTime(&v1);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = PetscTime(&v2);CHKERRQ(ierr);
  elapsed_time = v2 - v1;
  PetscPrintf(PETSC_COMM_WORLD,"\nelapsed time for symbolic and numerical factorization %g\n", elapsed_time);
  
  /* RHS vector assembling.*/
  ierr = VecCreateSeq(comm,n,&bv);CHKERRQ(ierr);
  ierr = VecDuplicate(bv,&xv);CHKERRQ(ierr);
  ierr = VecSet(bv, 2);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(bv);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(bv);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscTime(&v1);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,bv,xv);CHKERRQ(ierr);
  ierr = PetscTime(&v2);CHKERRQ(ierr);
  elapsed_time = v2 - v1;
  PetscPrintf(PETSC_COMM_WORLD,"\nelapsed time for backward/forward substitution %g\n", elapsed_time);
		
  /* RHS vectors assembling for computing rigid body modes.*/
  ierr = MatMumpsSetIcntl(F,25,-1);CHKERRQ(ierr);

  PetscInt  infog,num = 28;
  ierr = MatMumpsGetInfog(F,num,&infog);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(comm,n,infog,NULL,&b);CHKERRQ(ierr);
  ierr = MatDuplicate(b,MAT_DO_NOT_COPY_VALUES,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(b,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(b,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve for rigid body modes
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscTime(&v1);CHKERRQ(ierr);
  ierr = MatMatSolve(F,b,x);CHKERRQ(ierr);
  ierr = PetscTime(&v2);CHKERRQ(ierr);
  elapsed_time = v2 - v1;
  PetscPrintf(PETSC_COMM_WORLD,"\n elapsed time for solving for rigid body modes %g\n", elapsed_time);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Show results
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (elems < 5) {  
    PetscPrintf(PETSC_COMM_WORLD,"\nMatrix \n");
    MatView(A,PETSC_VIEWER_STDOUT_SELF);
		
    PetscPrintf(PETSC_COMM_WORLD,"\nRigid Body Modes \n");
    MatView(x,PETSC_VIEWER_STDOUT_SELF);

    PetscPrintf(PETSC_COMM_WORLD,"\nSolution Vector \n");
    VecView(xv,PETSC_VIEWER_STDOUT_SELF);
  }
		
  ierr = VecDestroy(&xv);CHKERRQ(ierr);
  ierr = VecDestroy(&bv);CHKERRQ(ierr);
  ierr = MatDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  /*
    Always call PetscFinalize() before exiting a program.  This routine
    - finalizes the PETSc libraries as well as MPI
    - provides summary and diagnostic information if certain runtime
    options are chosen (e.g., -log_summary).
  */
  ierr = PetscFinalize();
  return 0;
}
