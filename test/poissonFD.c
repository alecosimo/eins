static char help[] = "Solve the 2D Poisson equation using Finite Differences in a unit square.\n\n\
Exaple usage: \n\
mpiexec -n 4 ./poissonFD -npx 2 -npy 2 -nex 2 -ney 2\n\
Dirichlet boundaries on x=0 side by default. Options: 
-boundary: Dirichlet BC value\n\ 
-source: source value\n\ 
-npx,npy: subdomains per dimension\n\ 
-nex,ney: number of divisions per dimension\n\n";

#include <eins.h>
#include <private/einsfetiimpl.h> /* Users should never call include private includes. In this case we do it to access
				     members of the FETI object and test it correct behaviour*/
#include <petscblaslapack.h>
#define DEBUG 0

/* structure holding domain data */
typedef struct {
  /* communicator */
  MPI_Comm gcomm;
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
  DMBoundaryType         bx = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE, bz = DM_BOUNDARY_NONE;
  DMDAStencilType        stype = DMDA_STENCIL_BOX;
  ISLocalToGlobalMapping temp_isg2lmap;
  PetscInt               i,j,k,ig,jg,kg,lindex,gindex,localsize;
  PetscInt               *global_indices;

  PetscFunctionBeginUser;
  /* Not an efficient mapping: this function computes a very simple lexicographic mapping
     just to illustrate the creation of a MATIS object */
  localsize = dd.xm_l*dd.ym_l*dd.zm_l;
  ierr      = PetscMalloc1(localsize,&global_indices);CHKERRQ(ierr);
  for (k=0; k<dd.zm_l; k++) {
    kg=dd.startz+k;
    for (j=0; j<dd.ym_l; j++) {
      jg=dd.starty+j;
      for (i=0; i<dd.xm_l; i++) {
        ig                    =dd.startx+i;
        lindex                =k*dd.xm_l*dd.ym_l+j*dd.xm_l+i;
        gindex                =kg*dd.xm*dd.ym+jg*dd.xm+ig;
        global_indices[lindex]=gindex;
      }
    }
  }
  if (dd.dim==3) {
    ierr = DMDACreate3d(dd.gcomm,bx,by,bz,stype,dd.xm,dd.ym,dd.zm,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  } else if (dd.dim==2) {
    ierr = DMDACreate2d(dd.gcomm,bx,by,stype,dd.xm,dd.ym,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  } else {
    ierr = DMDACreate1d(dd.gcomm,bx,dd.xm,1,1,NULL,&da);CHKERRQ(ierr);
  }
  ierr = DMDASetAOType(da,AOMEMORYSCALABLE);CHKERRQ(ierr);
  ierr = DMDAGetAO(da,&ao);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,dd.xm_l*dd.ym_l*dd.zm_l,global_indices);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(dd.gcomm,1,localsize,global_indices,PETSC_OWN_POINTER,&temp_isg2lmap);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  *isg2lmap = temp_isg2lmap;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeSubdomainMatrix"
static PetscErrorCode ComputeSubdomainMatrix(DomainData dd, Mat *local_mat)
{
  PetscErrorCode ierr;
  PetscInt       localsize,yloc,xloc,auxnex,auxney;
  PetscInt       ie,je,ke,i,j,k,ig,jg,kg,ii,ming,I,J;
  PetscInt       *indexg,*cols,*colsg;
  PetscScalar    *vals,hx,hy;
  Mat            temp_local_mat;

  PetscFunctionBeginUser;

  /* Assemble subdomain matrix */
  localsize = dd.xm_l*dd.ym_l;
  ierr      = MatCreate(PETSC_COMM_SELF,&temp_local_mat);CHKERRQ(ierr);
  ierr      = MatSetSizes(temp_local_mat,localsize,localsize,localsize,localsize);CHKERRQ(ierr);

  ierr = MatSetType(temp_local_mat,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(temp_local_mat,1,5,NULL);CHKERRQ(ierr);      /* very overestimated */
  ierr = MatSetOption(temp_local_mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);


 for (I=Istart; I<Iend; I++) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if (i>0)   {J = I - n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = I + n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRQ(ierr);
  }


  
  yloc = 2;
  auxney = dd.ney_l;
  auxnex = dd.nex_l;

  for (ke=0; ke<auxnez; ke++) {
    for (je=0; je<auxney; je++) {
      for (ie=0; ie<auxnex; ie++) {
        xloc    = dd.p+1;
        ming    = 0;
        usedmat = &glldata.elem_mat;
        /* local to the element/global to the subdomain indexing */
        for (k=0; k<zloc; k++) {
          kg = ke*dd.p+k;
          for (j=0; j<yloc; j++) {
            jg = je*dd.p+j;
            for (i=0; i<xloc; i++) {
              ig         = ie*dd.p+i+ming;
              ii         = k*xloc*yloc+j*xloc+i;
              indexg[ii] = kg*dd.xm_l*dd.ym_l+jg*dd.xm_l+ig;
            }
          }
        }
        /* Set values */
        for (i=0; i<xloc*yloc*zloc; i++) {
          ierr = MatGetRow(*usedmat,i,&j,(const PetscInt**)&cols,(const PetscScalar**)&vals);CHKERRQ(ierr);
          for (k=0; k<j; k++) colsg[k] = indexg[cols[k]];
          ierr = MatSetValues(temp_local_mat,1,&indexg[i],j,colsg,vals,ADD_VALUES);CHKERRQ(ierr);
          ierr = MatRestoreRow(*usedmat,i,&j,(const PetscInt**)&cols,(const PetscScalar**)&vals);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(temp_local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (temp_local_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *local_mat = temp_local_mat;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "GLLStuffs"
static PetscErrorCode GLLStuffs(DomainData dd, GLLData *glldata)
{
  PetscErrorCode ierr;
  PetscReal      *M,si;
  PetscScalar    x,z0,z1,z2,Lpj,Lpr,rhoGLj,rhoGLk;
  PetscBLASInt   pm1,lierr;
  PetscInt       i,j,n,k,s,r,q,ii,jj,p=dd.p;
  PetscInt       xloc,yloc,zloc,xyloc,xyzloc;

  PetscFunctionBeginUser;
  /* Gauss-Lobatto-Legendre nodes zGL on [-1,1] */
  ierr = PetscMalloc1(p+1,&glldata->zGL);CHKERRQ(ierr);
  ierr = PetscMemzero(glldata->zGL,(p+1)*sizeof(*glldata->zGL));CHKERRQ(ierr);

  glldata->zGL[0]=-1.0;
  glldata->zGL[p]= 1.0;
  if (p > 1) {
    if (p == 2) glldata->zGL[1]=0.0;
    else {
      ierr = PetscMalloc1(p-1,&M);CHKERRQ(ierr);
      for (i=0; i<p-1; i++) {
        si  = (PetscReal)(i+1.0);
        M[i]=0.5*PetscSqrtReal(si*(si+2.0)/((si+0.5)*(si+1.5)));
      }
      pm1  = p-1;
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKsteqr",LAPACKsteqr_("N",&pm1,&glldata->zGL[1],M,&x,&pm1,M,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in STERF Lapack routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      ierr = PetscFree(M);CHKERRQ(ierr);
    }
  }

  /* Weights for 1D quadrature */
  ierr = PetscMalloc1(p+1,&glldata->rhoGL);CHKERRQ(ierr);

  glldata->rhoGL[0]=2.0/(PetscScalar)(p*(p+1.0));
  glldata->rhoGL[p]=glldata->rhoGL[0];
  z2 = -1;                      /* Dummy value to avoid -Wmaybe-initialized */
  for (i=1; i<p; i++) {
    x  = glldata->zGL[i];
    z0 = 1.0;
    z1 = x;
    for (n=1; n<p; n++) {
      z2 = x*z1*(2.0*n+1.0)/(n+1.0)-z0*(PetscScalar)(n/(n+1.0));
      z0 = z1;
      z1 = z2;
    }
    glldata->rhoGL[i]=2.0/(p*(p+1.0)*z2*z2);
  }

  /* Auxiliary mat for laplacian */
  ierr = PetscMalloc1(p+1,&glldata->A);CHKERRQ(ierr);
  ierr = PetscMalloc1((p+1)*(p+1),&glldata->A[0]);CHKERRQ(ierr);
  for (i=1; i<p+1; i++) glldata->A[i]=glldata->A[i-1]+p+1;

  for (j=1; j<p; j++) {
    x =glldata->zGL[j];
    z0=1.0;
    z1=x;
    for (n=1; n<p; n++) {
      z2=x*z1*(2.0*n+1.0)/(n+1.0)-z0*(PetscScalar)(n/(n+1.0));
      z0=z1;
      z1=z2;
    }
    Lpj=z2;
    for (r=1; r<p; r++) {
      if (r == j) {
        glldata->A[j][j]=2.0/(3.0*(1.0-glldata->zGL[j]*glldata->zGL[j])*Lpj*Lpj);
      } else {
        x  = glldata->zGL[r];
        z0 = 1.0;
        z1 = x;
        for (n=1; n<p; n++) {
          z2=x*z1*(2.0*n+1.0)/(n+1.0)-z0*(PetscScalar)(n/(n+1.0));
          z0=z1;
          z1=z2;
        }
        Lpr             = z2;
        glldata->A[r][j]=4.0/(p*(p+1.0)*Lpj*Lpr*(glldata->zGL[j]-glldata->zGL[r])*(glldata->zGL[j]-glldata->zGL[r]));
      }
    }
  }
  for (j=1; j<p+1; j++) {
    x  = glldata->zGL[j];
    z0 = 1.0;
    z1 = x;
    for (n=1; n<p; n++) {
      z2=x*z1*(2.0*n+1.0)/(n+1.0)-z0*(PetscScalar)(n/(n+1.0));
      z0=z1;
      z1=z2;
    }
    Lpj             = z2;
    glldata->A[j][0]=4.0*PetscPowRealInt(-1.0,p)/(p*(p+1.0)*Lpj*(1.0+glldata->zGL[j])*(1.0+glldata->zGL[j]));
    glldata->A[0][j]=glldata->A[j][0];
  }
  for (j=0; j<p; j++) {
    x  = glldata->zGL[j];
    z0 = 1.0;
    z1 = x;
    for (n=1; n<p; n++) {
      z2=x*z1*(2.0*n+1.0)/(n+1.0)-z0*(PetscScalar)(n/(n+1.0));
      z0=z1;
      z1=z2;
    }
    Lpj=z2;

    glldata->A[p][j]=4.0/(p*(p+1.0)*Lpj*(1.0-glldata->zGL[j])*(1.0-glldata->zGL[j]));
    glldata->A[j][p]=glldata->A[p][j];
  }
  glldata->A[0][0]=0.5+(p*(p+1.0)-2.0)/6.0;
  glldata->A[p][p]=glldata->A[0][0];

  /* compute element matrix */
  xloc = p+1;
  yloc = p+1;
  zloc = p+1;
  if (dd.dim<2) yloc=1;
  if (dd.dim<3) zloc=1;
  xyloc  = xloc*yloc;
  xyzloc = xloc*yloc*zloc;

  ierr = MatCreate(PETSC_COMM_SELF,&glldata->elem_mat);
  ierr = MatSetSizes(glldata->elem_mat,xyzloc,xyzloc,xyzloc,xyzloc);CHKERRQ(ierr);
  ierr = MatSetType(glldata->elem_mat,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(glldata->elem_mat,xyzloc,NULL);CHKERRQ(ierr); /* overestimated */
  ierr = MatZeroEntries(glldata->elem_mat);CHKERRQ(ierr);
  ierr = MatSetOption(glldata->elem_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  for (k=0; k<zloc; k++) {
    if (dd.dim>2) rhoGLk=glldata->rhoGL[k];
    else rhoGLk=1.0;

    for (j=0; j<yloc; j++) {
      if (dd.dim>1) rhoGLj=glldata->rhoGL[j];
      else rhoGLj=1.0;

      for (i=0; i<xloc; i++) {
        ii = k*xyloc+j*xloc+i;
        s  = k;
        r  = j;
        for (q=0; q<xloc; q++) {
          jj   = s*xyloc+r*xloc+q;
          ierr = MatSetValue(glldata->elem_mat,jj,ii,glldata->A[i][q]*rhoGLj*rhoGLk,ADD_VALUES);CHKERRQ(ierr);
        }
        if (dd.dim>1) {
          s=k;
          q=i;
          for (r=0; r<yloc; r++) {
            jj   = s*xyloc+r*xloc+q;
            ierr = MatSetValue(glldata->elem_mat,jj,ii,glldata->A[j][r]*glldata->rhoGL[i]*rhoGLk,ADD_VALUES);CHKERRQ(ierr);
          }
        }
        if (dd.dim>2) {
          r=j;
          q=i;
          for (s=0; s<zloc; s++) {
            jj   = s*xyloc+r*xloc+q;
            ierr = MatSetValue(glldata->elem_mat,jj,ii,glldata->A[k][s]*rhoGLj*glldata->rhoGL[i],ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = MatAssemblyBegin(glldata->elem_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (glldata->elem_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if DEBUG
  {
    Vec       lvec,rvec;
    PetscReal norm;
    ierr = MatCreateVecs(glldata->elem_mat,&lvec,&rvec);CHKERRQ(ierr);
    ierr = VecSet(lvec,1.0);CHKERRQ(ierr);
    ierr = MatMult(glldata->elem_mat,lvec,rvec);CHKERRQ(ierr);
    ierr = VecNorm(rvec,NORM_INFINITY,&norm);CHKERRQ(ierr);
    printf("Test null space of elem mat % 1.14e\n",norm);
    ierr = VecDestroy(&lvec);CHKERRQ(ierr);
    ierr = VecDestroy(&rvec);CHKERRQ(ierr);
  }
#endif
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
  dd->xm_l = dd->nex_l*dd->p+1;
  dd->xm   = dd->nex*dd->p+1;
  dd->ym_l = dd->ney_l*dd->p+1;
  dd->ym   = dd->ney*dd->p+1;

  /* starting global index for local dofs (simple lexicographic order) */
  dd->startx = 0;
  j          = dd->nex/dd->npx;
  for (i=0; i<dd->ipx; i++) {
    k = j;
    if (i<dd->nex%dd->npx) k++;
    dd->startx = dd->startx+k*dd->p;
  }

  dd->starty = 0;
  if (dd->dim > 1) {
    j = dd->ney/dd->npy;
    for (i=0; i<dd->ipy; i++) {
      k = j;
      if (i<dd->ney%dd->npy) k++;
      dd->starty = dd->starty+k*dd->p;
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeDirichletLocalRows"
static PetscErrorCode ComputeDirichletLocalRows(DomainData dd,PetscInt **dirichlet,PetscInt *n_dirichlet)
{
  PetscErrorCode ierr;
  PetscInt       localsize=0,i,j,k,*indices=0;

  PetscFunctionBeginUser; 
  if (dd.ipx == 0) {    /* west boundary */
    localsize = dd.ym_l*dd.zm_l;
    ierr = PetscMalloc1(localsize,&indices);CHKERRQ(ierr);
    i = 0;

    for (k=0;k<dd.zm_l;k++)
      for (j=0;j<dd.ym_l;j++)
	indices[i++]=k*dd.ym_l*dd.xm_l+j*dd.xm_l;
  }
  *dirichlet = indices;
  *n_dirichlet = localsize;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrixAndRHS"
static PetscErrorCode ComputeMatrixAndRHS(DomainData dd,Mat* localA,Vec* localRHS,PetscScalar bcond,PetscScalar source)
{
  PetscErrorCode         ierr;
  PetscInt               localsize,*dirichlet,n_dirichlet;
  Vec                    tempRHS,vfix;
  
  PetscFunctionBeginUser;
  /* Compute matrix of subdomain Neumann problem */
  ierr = ComputeSubdomainMatrix(dd,localA);CHKERRQ(ierr);
  /* Compute RHS */
  ComputeDirichletLocalRows(dd,&dirichlet,&n_dirichlet);
  localsize = dd.xm_l*dd.ym_l*dd.zm_l;
  ierr      = VecCreateSeq(PETSC_COMM_SELF,localsize,&tempRHS);CHKERRQ(ierr);
  ierr      = VecSet(tempRHS,source);CHKERRQ(ierr);
  ierr      = VecAssemblyBegin(tempRHS);CHKERRQ(ierr);
  ierr      = VecAssemblyEnd(tempRHS);CHKERRQ(ierr);
  //MatSeqViewSynchronized(dd.gcomm,*localA);
  if (n_dirichlet) {
    ierr      = MatSetOption(*localA,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr      = MatSetOption(*localA,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
    ierr      = VecDuplicate(tempRHS,&vfix);CHKERRQ(ierr);
    ierr      = VecSet(vfix,bcond);CHKERRQ(ierr);
    ierr      = VecAssemblyBegin(vfix);CHKERRQ(ierr);
    ierr      = VecAssemblyEnd(vfix);CHKERRQ(ierr);
    ierr      = MatSetOption(*localA,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);

    ierr      = MatZeroRowsColumns(*localA,n_dirichlet,dirichlet,1.0,vfix,tempRHS);CHKERRQ(ierr);
  } else {
    ierr = MatSetOption(*localA,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(*localA,MAT_SPD,PETSC_FALSE);CHKERRQ(ierr);
  }
  *localRHS = tempRHS;
  /* free allocated workspace */
  ierr = PetscFree(gll.zGL);CHKERRQ(ierr);
  ierr = PetscFree(gll.rhoGL);CHKERRQ(ierr);
  ierr = PetscFree(gll.A[0]);CHKERRQ(ierr);
  ierr = PetscFree(gll.A);CHKERRQ(ierr);
  ierr = PetscFree(dirichlet);CHKERRQ(ierr);
  ierr = MatDestroy(&gll.elem_mat);CHKERRQ(ierr);
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
  ierr    = PetscOptionsGetInt (NULL,"-npx",&dd->npx,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetInt (NULL,"-npy",&dd->npy,NULL);CHKERRQ(ierr);
  /* Number of elements per dimension */
  /* Default is one element per subdomain */
  dd->nex = dd->npx;
  dd->ney = dd->npy;
  ierr    = PetscOptionsGetInt (NULL,"-nex",&dd->nex,NULL);CHKERRQ(ierr);
  ierr    = PetscOptionsGetInt (NULL,"-ney",&dd->ney,NULL);CHKERRQ(ierr);
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
  Mat                      localA=0;
  Vec                      localRHS=0,u_local=0,global_sol=0;
  ISLocalToGlobalMapping   mapping=0;
  FETI                     feti;
  KSP                      ksp_interface;
  PetscScalar              boundary=-5,source=-2;
  
  /* Init EINS */
  EinsInitialize(&argc,&args,(char*)0,help);
  /* Initialize DomainData */
  ierr = InitializeDomainData(&dd);CHKERRQ(ierr);
  /* Decompose domain */
  ierr = DomainDecomposition(&dd);CHKERRQ(ierr);
  /* assemble global matrix */
  ierr = PetscOptionsGetScalar (NULL,"-boundary",&boundary,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar (NULL,"-source",&source,NULL);CHKERRQ(ierr);
  ierr = ComputeMatrixAndRHS(dd,&localA,&localRHS,boundary,source);CHKERRQ(ierr);
  /* Compute global mapping of local dofs */
  ierr = ComputeMapping(dd,&mapping);CHKERRQ(ierr);
  
  ierr = VecDuplicate(localRHS,&u_local);CHKERRQ(ierr);
  /* Setting FETI */
  ierr = FETICreate(dd.gcomm,&feti);CHKERRQ(ierr);
  ierr = FETISetType(feti,FETI1);CHKERRQ(ierr);
  ierr = FETI1SetDefaultOptions(&argc,&args,NULL);CHKERRQ(ierr);
  ierr = FETISetFromOptions(feti);CHKERRQ(ierr);
  ierr = FETISetLocalMat(feti,localA);CHKERRQ(ierr);
  ierr = FETISetLocalRHS(feti,localRHS);CHKERRQ(ierr);
  ierr = FETISetInterfaceSolver(feti,KSPPJCG,PCFETI_DIRICHLET);CHKERRQ(ierr);//
  ierr = FETISetMapping(feti,mapping);CHKERRQ(ierr);
  ierr = ISCreateMPIVec(dd.gcomm,dd.xm*dd.ym*dd.zm,mapping,&global_sol);CHKERRQ(ierr);
  ierr = FETISetGlobalSolutionVector(feti,global_sol);CHKERRQ(ierr);
  ierr = FETISetUp(feti);CHKERRQ(ierr);
  ierr = FETIGetKSPInterface(feti,&ksp_interface);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp_interface,1e-10,0,PETSC_DEFAULT,1000);CHKERRQ(ierr);
  ierr = FETISolve(feti,u_local);CHKERRQ(ierr);

  ierr = VecSeqViewSynchronized(dd.gcomm,u_local);CHKERRQ(ierr);
  
  ierr = FETIDestroy(&feti);CHKERRQ(ierr);
  ierr = MatDestroy(&localA);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_interface);CHKERRQ(ierr);
  ierr = VecDestroy(&u_local);CHKERRQ(ierr);
  ierr = VecDestroy(&localRHS);CHKERRQ(ierr);
  ierr = VecDestroy(&global_sol);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);
  ierr = EinsFinalize();CHKERRQ(ierr);

  return 0;
}

