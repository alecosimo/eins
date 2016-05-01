static char help[] = "Solve the 2D Poisson equation using Finite Differences in a cavity of lx*ly.\n\n\
Exaple usage:\n\
mpiexec -n 4 ./poissonFD -npx 2 -npy 2 -nex 2 -ney 2\n\
Dirichlet boundaries on x=0 side by default. Options:\n\
-boundary: Dirichlet BC value\n\
-source: source value\n\
-npx,npy: subdomains per dimension\n\
-nex,ney: number of divisions per dimension\n\
-lx,ly: length of the cavity in the x and y directions\n\n";

#include <eins.h>

/* structure holding domain data */
typedef struct {
  /* communicator */
  MPI_Comm gcomm;
  /* Dirichlet BC value, source value */
  PetscScalar source, boundary;
  /* length of the cavity in the x and y directions */
  PetscScalar lx,ly,hx,hy;
  /* subdomains per dimension */
  PetscInt npx,npy;
  /* subdomain index in cartesian dimensions */
  PetscInt ipx,ipy;
  /* elements per dimension */
  PetscInt nex,ney;
  /* local elements per dimension */
  PetscInt nex_l,ney_l;
  /* global number of dofs per dimension */
  PetscInt xm,ym;
  /* local number of dofs per dimension */
  PetscInt xm_l,ym_l;
  /* starting global indexes for subdomain in lexicographic ordering */
  PetscInt startx,starty,startz;
} DomainData;


#undef __FUNCT__
#define __FUNCT__ "ComputeMapping"
static PetscErrorCode ComputeMapping(DomainData dd,ISLocalToGlobalMapping *isg2lmap)
{
  PetscErrorCode         ierr;
  DM                     da;
  AO                     ao;
  DMBoundaryType         bx = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType        stype = DMDA_STENCIL_BOX;
  ISLocalToGlobalMapping temp_isg2lmap;
  PetscInt               i,j,ig,jg,lindex,gindex,localsize;
  PetscInt               *global_indices;

  PetscFunctionBeginUser;
  /* Not an efficient mapping: this function computes a very simple lexicographic mapping
     just to illustrate the creation of a MATIS object */
  localsize = dd.xm_l*dd.ym_l;
  ierr      = PetscMalloc1(localsize,&global_indices);CHKERRQ(ierr);
  for (j=0; j<dd.ym_l; j++) {
    jg=dd.starty+j;
    for (i=0; i<dd.xm_l; i++) {
      ig                    =dd.startx+i;
      lindex                =j*dd.xm_l+i;
      gindex                =jg*dd.xm+ig;
      global_indices[lindex]=gindex;
    }
  }
  ierr = DMDACreate2d(dd.gcomm,bx,by,stype,dd.xm,dd.ym,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetAOType(da,AOMEMORYSCALABLE);CHKERRQ(ierr);
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,dd.xm_l*dd.ym_l,global_indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(dd.gcomm,1,localsize,global_indices,PETSC_OWN_POINTER,&temp_isg2lmap);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  *isg2lmap = temp_isg2lmap;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DomainDecomposition"
static PetscErrorCode DomainDecomposition(DomainData *dd)
{
  PetscMPIInt rank;
  PetscInt    i,j,k;

  PetscFunctionBeginUser;
  /* Subdomain index in cartesian coordinates */
  MPI_Comm_rank(dd->gcomm,&rank);
  dd->ipx = rank%dd->npx;
  dd->ipy = rank/dd->npx-rank/(dd->npx*dd->npy)*dd->npy;
  /* number of local elements */
  dd->nex_l = dd->nex/dd->npx;
  if (dd->ipx < dd->nex%dd->npx) dd->nex_l++;
  dd->ney_l = dd->ney/dd->npy;
  if (dd->ipy < dd->ney%dd->npy) dd->ney_l++;
  /* local and global number of dofs */
  dd->xm_l = dd->nex_l+1;
  dd->xm   = dd->nex+1;
  dd->ym_l = dd->ney_l+1;
  dd->ym   = dd->ney+1;

  /* starting global index for local dofs (simple lexicographic order) */
  dd->startx = 0;
  j          = dd->nex/dd->npx;
  for (i=0; i<dd->ipx; i++) {
    k = j;
    if (i<dd->nex%dd->npx) k++;
    dd->startx = dd->startx+k;
  }

  dd->starty = 0;
  j = dd->ney/dd->npy;
  for (i=0; i<dd->ipy; i++) {
    k = j;
    if (i<dd->ney%dd->npy) k++;
    dd->starty = dd->starty+k;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeDirichletLocalRows"
static PetscErrorCode ComputeDirichletLocalRows(DomainData dd,PetscInt **dirichlet,PetscInt *n_dirichlet)
{
  PetscErrorCode ierr;
  PetscInt       localsize=0,j,*indices=0;

  PetscFunctionBeginUser; 
  if (dd.ipx == 0) {    /* west boundary */
    localsize = dd.ym_l;
    ierr = PetscMalloc1(localsize,&indices);CHKERRQ(ierr);
    for (j=0;j<dd.ym_l;j++)
      indices[j]=j*dd.xm_l;
  }
  *dirichlet = indices;
  *n_dirichlet = localsize;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrixAndRHS"
static PetscErrorCode ComputeMatrixAndRHS(DomainData dd,Mat* localA,Vec* localRHS)
{
  PetscErrorCode         ierr;
  PetscInt               localsize,*dirichlet=0,n_dirichlet=0;
  Vec                    tempRHS,vfix;
  PetscInt               n,m,i,j,Ii,Jj;
  PetscScalar            hx,hy;
  Mat                    temp_local_mat;
  
  PetscFunctionBeginUser;
  localsize = dd.xm_l*dd.ym_l;
  ierr      = VecCreateSeq(PETSC_COMM_SELF,localsize,&tempRHS);CHKERRQ(ierr);
  ierr      = VecSet(tempRHS,dd.source);CHKERRQ(ierr);
  /* Assemble subdomain matrix */
  ierr      = MatCreate(PETSC_COMM_SELF,&temp_local_mat);CHKERRQ(ierr);
  ierr      = MatSetSizes(temp_local_mat,localsize,localsize,localsize,localsize);CHKERRQ(ierr);

  ierr = MatSetType(temp_local_mat,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(temp_local_mat,1,5,NULL);CHKERRQ(ierr);      /* very overestimated */
  ierr = MatSetOption(temp_local_mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);

  n  = dd.xm_l; m  = dd.ym_l;
  hx = dd.hx;   hy = dd.hy;
  for (Ii=0; Ii<localsize; Ii++) { 
    j = Ii/n;     i = Ii - j*n;
    if (i==n-1) {
      if (j<m-1) {Jj = Ii + n; ierr = MatSetValue(temp_local_mat,Ii,Jj,-0.5/hy/hy,INSERT_VALUES);CHKERRQ(ierr);}
      if ((j==0)||(j==m-1)){
	ierr = MatSetValue(temp_local_mat,Ii,Ii,0.5/hx/hx + 0.5/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecSetValue(tempRHS,Ii,dd.source/4.0,INSERT_VALUES);CHKERRQ(ierr);
      }else{
	ierr = MatSetValue(temp_local_mat,Ii,Ii,1.0/hx/hx + 1.0/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecSetValue(tempRHS,Ii,dd.source/2.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    } else if ((i==0)&&(dd.ipx > 0)) {
      if (j<m-1) {Jj = Ii + n; ierr = MatSetValue(temp_local_mat,Ii,Jj,-0.5/hy/hy,INSERT_VALUES);CHKERRQ(ierr);}
      if ((j==0)||(j==m-1)){
	Jj = Ii + 1; ierr = MatSetValue(temp_local_mat,Ii,Jj,-0.5/hx/hx,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatSetValue(temp_local_mat,Ii,Ii,0.5/hx/hx + 0.5/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecSetValue(tempRHS,Ii,dd.source/4.0,INSERT_VALUES);CHKERRQ(ierr);
      }else{
	Jj = Ii + 1; ierr = MatSetValue(temp_local_mat,Ii,Jj,-1.0/hx/hx,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatSetValue(temp_local_mat,Ii,Ii,1.0/hx/hx + 1.0/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecSetValue(tempRHS,Ii,dd.source/2.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    } else if ((j==0)&&(i>0)&&(i<n-1)) {
      Jj = Ii + n; ierr = MatSetValue(temp_local_mat,Ii,Jj,-1.0/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
      if (i<n-1) {Jj = Ii + 1; ierr = MatSetValue(temp_local_mat,Ii,Jj,-0.5/hx/hx,INSERT_VALUES);CHKERRQ(ierr);}
      ierr = MatSetValue(temp_local_mat,Ii,Ii,1.0/hx/hx + 1.0/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(tempRHS,Ii,dd.source/2.0,INSERT_VALUES);CHKERRQ(ierr);
    } else if ((j==m-1)&&(i>0)&&(i<n-1)) {
      if (i<n-1) {Jj = Ii + 1; ierr = MatSetValue(temp_local_mat,Ii,Jj,-0.5/hx/hx,INSERT_VALUES);CHKERRQ(ierr);}
      ierr = MatSetValue(temp_local_mat,Ii,Ii,1.0/hx/hx + 1.0/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(tempRHS,Ii,dd.source/2.0,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      if (j<m-1) {Jj = Ii + n; ierr = MatSetValue(temp_local_mat,Ii,Jj,-1.0/hy/hy,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<n-1) {Jj = Ii + 1; ierr = MatSetValue(temp_local_mat,Ii,Jj,-1.0/hx/hx,INSERT_VALUES);CHKERRQ(ierr);}
      ierr = MatSetValue(temp_local_mat,Ii,Ii,2.0/hx/hx + 2.0/hy/hy,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr    = MatAssemblyBegin(temp_local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr    = MatAssemblyEnd  (temp_local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *localA = temp_local_mat;
  ierr    = VecAssemblyBegin(tempRHS);CHKERRQ(ierr);
  ierr    = VecAssemblyEnd(tempRHS);CHKERRQ(ierr);

  /* Compute RHS */
  ComputeDirichletLocalRows(dd,&dirichlet,&n_dirichlet);
  //MatSeqViewSynchronized(dd.gcomm,*localA);
  if (n_dirichlet) {
    ierr      = MatSetOption(*localA,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr      = MatSetOption(*localA,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
    ierr      = VecDuplicate(tempRHS,&vfix);CHKERRQ(ierr);
    ierr      = VecSet(vfix,dd.boundary);CHKERRQ(ierr);
    ierr      = VecAssemblyBegin(vfix);CHKERRQ(ierr);
    ierr      = VecAssemblyEnd(vfix);CHKERRQ(ierr);
    ierr      = MatSetOption(*localA,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);

    ierr      = MatZeroRowsColumns(*localA,n_dirichlet,dirichlet,1.0,vfix,tempRHS);CHKERRQ(ierr);
    ierr      = VecDestroy(&vfix);CHKERRQ(ierr);
  } else {
    ierr = MatSetOption(*localA,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(*localA,MAT_SPD,PETSC_FALSE);CHKERRQ(ierr);
  }
  *localRHS = tempRHS;
  /* free allocated workspace */
  if(dirichlet) { ierr = PetscFree(dirichlet);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "InitializeDomainData"
static PetscErrorCode InitializeDomainData(DomainData *dd)
{
  PetscErrorCode ierr;
  PetscMPIInt    sizes,rank;

  PetscFunctionBeginUser;
  dd->gcomm = PETSC_COMM_WORLD;
  ierr      = MPI_Comm_size(dd->gcomm,&sizes);
  ierr      = MPI_Comm_rank(dd->gcomm,&rank);
  /* test data passed in */
  if (sizes<2) SETERRQ(dd->gcomm,PETSC_ERR_USER,"This is not a uniprocessor test");
  /* Get informations from command line */
  /* Processors/subdomains per dimension */
  /* Default is 1d problem */
  dd->npx = sizes/2+sizes%2;
  dd->npy = sizes/2;
  ierr    = PetscOptionsGetInt (NULL,NULL,"-npx",&dd->npx,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetInt (NULL,NULL,"-npy",&dd->npy,NULL);CHKERRQ(ierr);
  /* Number of elements per dimension */
  /* Default is one element per subdomain */
  dd->nex = dd->npx;
  dd->ney = dd->npy;
  ierr    = PetscOptionsGetInt (NULL,NULL,"-nex",&dd->nex,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetInt (NULL,NULL,"-ney",&dd->ney,NULL);CHKERRQ(ierr);
  dd->lx  = dd->npx;
  dd->ly  = dd->npy;
  ierr    = PetscOptionsGetScalar (NULL,NULL,"-ly",&dd->ly,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetScalar (NULL,NULL,"-lx",&dd->lx,NULL);CHKERRQ(ierr);
  dd->hx  = dd->lx/dd->nex;
  dd->hy  = dd->ly/dd->ney;
  dd->boundary = -5.0;
  dd->source   = -2.0;
  ierr = PetscOptionsGetScalar (NULL,NULL,"-boundary",&dd->boundary,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar (NULL,NULL,"-source",&dd->source,NULL);CHKERRQ(ierr);

  if (sizes!=dd->npx*dd->npy) SETERRQ(dd->gcomm,PETSC_ERR_USER,"Number of mpi procs in 2D must be equal to npx*npy");
  if (dd->nex<dd->npx || dd->ney<dd->npy) SETERRQ(dd->gcomm,PETSC_ERR_USER,"Number of elements per dim must be greater/equal than number of procs per dim");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode           ierr;
  DomainData               dd;
  /* PetscReal                norm,maxeig,mineig;*/
  Mat                      localA=0,lgmat=0;
  Vec                      localRHS=0,u=0;
  ISLocalToGlobalMapping   mapping=0;
  FETI                     feti;
  KSP                      ksp_interface;
  PetscInt                 rank,lsize;
  
  /* Init EINS */
  EinsInitialize(&argc,&args,(char*)0,help);
  /* Initialize DomainData */
  ierr = InitializeDomainData(&dd);CHKERRQ(ierr);
  /* Decompose domain */
  ierr = DomainDecomposition(&dd);CHKERRQ(ierr);
  /* assemble global matrix */
  lsize = dd.xm_l*dd.ym_l;
  ierr  = ComputeMatrixAndRHS(dd,&localA,&localRHS);CHKERRQ(ierr);
  ierr  = MatCreateLGMat(PETSC_COMM_WORLD,lsize,lsize,localA,&lgmat);CHKERRQ(ierr);
  /* Compute global mapping of local dofs */
  ierr = ComputeMapping(dd,&mapping);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(dd.gcomm,&rank);CHKERRQ(ierr);
  /* MatSeqViewSynchronized(PETSC_COMM_WORLD,localA); */
  /* VecSeqViewSynchronized(PETSC_COMM_WORLD,localRHS); */
  /* if(rank==2) */
  /* ISLocalToGlobalMappingView(mapping,PETSC_VIEWER_STDOUT_SELF); */

  /* Create u */
  ierr = VecCreate(dd.gcomm,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,lsize,dd.xm*dd.ym);CHKERRQ(ierr);
  ierr = VecSetType(u,VECMPIUNASM);CHKERRQ(ierr);
  
  /* Setting FETI */
  ierr = FETICreate(dd.gcomm,&feti);CHKERRQ(ierr);
  ierr = FETISetType(feti,FETI1);CHKERRQ(ierr);
  ierr = FETI1SetDefaultOptions(&argc,&args,NULL);CHKERRQ(ierr);
  ierr = FETISetFromOptions(feti);CHKERRQ(ierr);
  ierr = FETISetMat(feti,lgmat);CHKERRQ(ierr);
  ierr = FETISetLocalRHS(feti,localRHS);CHKERRQ(ierr);
  ierr = FETISetInterfaceSolver(feti,KSPPJCG,PCFETI_DIRICHLET);CHKERRQ(ierr);//
  ierr = FETISetMappingAndGlobalSize(feti,mapping,dd.xm*dd.ym);CHKERRQ(ierr);

  ierr = FETISetUp(feti);CHKERRQ(ierr);
  ierr = FETIGetKSPInterface(feti,&ksp_interface);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp_interface,1e-10,0,PETSC_DEFAULT,1000);CHKERRQ(ierr);
  ierr = FETISolve(feti,u);CHKERRQ(ierr);

  if (dd.nex<10) {
#if PETSC_VERSION_LT(3,7,0)
    ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
#else
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
#endif
    ierr = VecSeqViewSynchronized(dd.gcomm,u);CHKERRQ(ierr);
  }

  ierr = FETIDestroy(&feti);CHKERRQ(ierr);
  ierr = MatDestroy(&localA);CHKERRQ(ierr);
  ierr = MatDestroy(&lgmat);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&localRHS);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);
  ierr = EinsFinalize();CHKERRQ(ierr);

  return 0;
}

