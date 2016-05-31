#include <../src/feti/einspj2level.h>
#include <../src/pc/einspcdirichlet.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einssys.h>
#include <einspc.h>


PetscErrorCode FETIPJProject_PJ2LEVEL(void*,Vec,Vec);
PetscErrorCode FETIPJReProject_PJ2LEVEL(void*,Vec,Vec);


#undef __FUNCT__
#define __FUNCT__ "FETIPJGatherNeighborsCoarseBasis_PJ2LEVEL"
static PetscErrorCode FETIPJGatherNeighborsCoarseBasis_PJ2LEVEL(FETIPJ ftpj)
{
  PetscErrorCode    ierr;
  FETI              ft  = ftpj->feti;
  PJ2LEVEL          *pj = (PJ2LEVEL*)ftpj->data;
  PetscMPIInt       i_mpi;
  PetscInt          i,j,n_cs,idx;
  PetscScalar       **array=NULL;
  MPI_Comm          comm;
  IS                isindex;
  Mat               *submat=NULL;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  /* Communicate matrices G */
  if(pj->n_send) {
    ierr = PetscMalloc1(pj->n_send,&submat);CHKERRQ(ierr);
    ierr = PetscMalloc1(pj->n_send,&array);CHKERRQ(ierr);
    for (j=0, i=1; i<ft->n_neigh_lb; i++){
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(ft->localG,isindex,NULL,MAT_INITIAL_MATRIX,&submat[j]);CHKERRQ(ierr);
      ierr = MatDenseGetArray(submat[j],&array[j]);CHKERRQ(ierr);   
      ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
      ierr = MPI_Isend(array[j],ft->n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&pj->send_reqs[j]);CHKERRQ(ierr);
      ierr = ISDestroy(&isindex);CHKERRQ(ierr);
      j++;
    }
  }
  if(pj->n_recv) {
    for (j=0,idx=0,i=1; i<ft->n_neigh_lb; i++){
      n_cs = pj->n_cs_comm[ft->neigh_lb[i]];
      if (n_cs>0) {
	ierr  = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
	ierr  = MPI_Irecv(&pj->matrices[idx],n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,0,comm,&pj->recv_reqs[j]);CHKERRQ(ierr);    
	idx  += n_cs*ft->n_shared_lb[i]; 
	j++;
      }	  
    }
  }
  if(pj->n_recv) {ierr = MPI_Waitall(pj->n_recv,pj->recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
  if(pj->n_send) {
    ierr = MPI_Waitall(pj->n_send,pj->send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    for (i=0;i<pj->n_send;i++) {
      ierr = MatDenseRestoreArray(submat[i],&array[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&submat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(array);CHKERRQ(ierr);
    ierr = PetscFree(submat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJAssembleCoarseProblem_PJ2LEVEL"
static PetscErrorCode FETIPJAssembleCoarseProblem_PJ2LEVEL(FETIPJ ftpj)
{
  PetscErrorCode    ierr;
  FETI              ft  = ftpj->feti;
  PJ2LEVEL          *pj = (PJ2LEVEL*)ftpj->data;
  Subdomain         sd = ft->subdomain;
  PetscMPIInt       i_mpi0,i_mpi1,sizeG,rankG;
  MPI_Comm          comm;
  IS                isindex;
  Mat               RHS,X,x,Gexpanded,*submat,aux_mat;
  PetscScalar       *pointer_vec2=NULL,*pointer_vec1=NULL,*m_pointer=NULL,*m_pointer1=NULL,**array=NULL;
  Vec               vec1,vec2;
  PetscInt          i,j,k,k0,idx,jdx,kdx,*idxm=NULL,*idxn=NULL;
  PetscInt          n_cs,delta,sz;
  
  PetscFunctionBegin;
  /* in the following rankG and sizeG are related to the MPI_Comm comm */
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rankG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&sizeG);CHKERRQ(ierr);
  
  /* computing F_local*G_neighbors */
  ierr = PetscMalloc1(pj->max_n_cs,&idxn);CHKERRQ(ierr);
  for (i=0;i<pj->max_n_cs;i++) idxn[i]=i;

  for (i=0,k=0; k<ft->n_neigh_lb; k++) {
    n_cs = pj->n_cs_comm[ft->neigh_lb[k]];
    if (n_cs>0) {
      /* the following matrix is created using in column major order (the usual Fortran 77 manner) */
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,n_cs,pj->bufferX,&X);CHKERRQ(ierr);
      ierr = MatSetOption(X,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,sd->n,n_cs,pj->bufferRHS,&RHS);CHKERRQ(ierr);
      ierr = MatSetOption(RHS,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      if (k>0) {
	ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,n_cs,pj->bufferG,&Gexpanded);CHKERRQ(ierr);
	ierr = MatSetOption(Gexpanded,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
	ierr = MatDenseGetArray(pj->Gholder[i-!(ft->n_cs==0)],&m_pointer);CHKERRQ(ierr);
	ierr = MatZeroEntries(Gexpanded);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(Gexpanded,ft->n_shared_lb[k],ft->shared_lb[k],n_cs,idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(Gexpanded,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(Gexpanded,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(pj->Gholder[i-!(ft->n_cs==0)],&m_pointer);CHKERRQ(ierr);
      } else {
	Gexpanded = ft->localG;
      }
	
      /**** RHS = B^T*Gexpanded */
      ierr = MatDenseGetArray(Gexpanded,&pointer_vec2);CHKERRQ(ierr);
      ierr = MatDenseGetArray(RHS,&pointer_vec1);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,ft->n_lambda_local,NULL,&vec2);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,sd->n,NULL,&vec1);CHKERRQ(ierr);
      for (j=0;j<n_cs;j++) {
	ierr = VecPlaceArray(vec2,(const PetscScalar*)(pointer_vec2+ft->n_lambda_local*j));CHKERRQ(ierr);
	ierr = VecPlaceArray(vec1,(const PetscScalar*)(pointer_vec1+sd->n*j));CHKERRQ(ierr);
	ierr = MatMultTranspose(ft->B_delta,vec2,sd->vec1_B);CHKERRQ(ierr);
	ierr = VecSet(vec1,0);CHKERRQ(ierr);
	ierr = VecScatterBegin(sd->N_to_B,sd->vec1_B,vec1,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
	ierr = VecScatterEnd(sd->N_to_B,sd->vec1_B,vec1,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
	ierr = VecResetArray(vec2);CHKERRQ(ierr);
	ierr = VecResetArray(vec1);CHKERRQ(ierr);
      }   
      ierr = MatDenseRestoreArray(Gexpanded,&pointer_vec2);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(RHS,&pointer_vec1);CHKERRQ(ierr);
      ierr = VecDestroy(&vec2);CHKERRQ(ierr);
      ierr = VecDestroy(&vec1);CHKERRQ(ierr);

      /**** solve system Kt*X = RHS */
      ierr = MatMatSolve(ft->F_neumann,RHS,X);CHKERRQ(ierr);

      /****  compute B*X */
      ierr = MatGetSubMatrix(X,sd->is_B_local,NULL,MAT_INITIAL_MATRIX,&x);CHKERRQ(ierr);
      ierr = MatMatMult(ft->B_delta,x,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pj->FGholder[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&x);CHKERRQ(ierr);
      ierr = MatDestroy(&RHS);CHKERRQ(ierr);
      ierr = MatDestroy(&X);CHKERRQ(ierr);
      
      if (k>0) { ierr = MatDestroy(&Gexpanded);CHKERRQ(ierr); }
      i++;
    }
  }

  /* communicate computed FG */
  if(pj->n_send2) { ierr = PetscMalloc2(pj->n_send2,&submat,pj->n_send2,&array);CHKERRQ(ierr);}
  for (delta=0,idx=0,jdx=0,i=1;i<ft->n_neigh_lb;i++) {
    kdx  = 0;
    k0   = ft->neigh_lb[i];
    ierr = PetscMPIIntCast(k0,&i_mpi0);CHKERRQ(ierr);
    /* I send */
    if (pj->n_send2) {
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
      if (ft->n_cs) {
	if (pj->n_cs_comm[k0]>0 && k0<rankG) {/*send myself Fs*Gs*/
	  /*>>>*/
	  ierr = MatGetSubMatrix(pj->FGholder[kdx],isindex,NULL,MAT_INITIAL_MATRIX,&submat[idx]);CHKERRQ(ierr);
	  ierr = MatDenseGetArray(submat[idx],&array[idx]);CHKERRQ(ierr);   
	  ierr = MPI_Isend(array[idx],ft->n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi0,rankG,comm,&pj->send2_reqs[idx]);CHKERRQ(ierr);
	  idx++;
	  /*<<<*/
	}
      	kdx++;
      }
      for (j=1;j<ft->n_neigh_lb;j++) { /*send Fs*G_{my_neighs}U{k0}*/
	k = ft->neigh_lb[j];
	if (pj->n_cs_comm[k]>0) {
	  if(k0<=k) {
	    /*>>>*/
	    ierr = MatGetSubMatrix(pj->FGholder[kdx],isindex,NULL,MAT_INITIAL_MATRIX,&submat[idx]);CHKERRQ(ierr);
	    ierr = MatDenseGetArray(submat[idx],&array[idx]);CHKERRQ(ierr);   
	    ierr = MPI_Isend(array[idx],pj->n_cs_comm[k]*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi0,k,comm,&pj->send2_reqs[idx]);CHKERRQ(ierr);
	    /*<<<*/
	    idx++;
	  }
	  kdx++;
	}
      }
      ierr = ISDestroy(&isindex);CHKERRQ(ierr);
    }
    /* I receive */   
    if(pj->n_recv2) {
      for (j=0;j<pj->n_neighs2[i-1];j++) { /*receive F_{my_neighs}*G_{my_neighs}U{neighs_neighs}*/
	k     = pj->neighs2[i-1][j];
	n_cs = pj->n_cs_comm[k];
	if(rankG<=k && n_cs) {
	  /*>>>*/
	  sz     = n_cs*ft->n_shared_lb[i];
	  ierr   = MPI_Irecv(&pj->fgmatrices[delta],sz,MPIU_SCALAR,i_mpi0,k,comm,&pj->recv2_reqs[jdx]);CHKERRQ(ierr);    
	  delta += sz; 
	  jdx++;
	  /*<<<*/
	}
      }
    }
  }
  if(pj->n_recv2) {
    ierr = MPI_Waitall(pj->n_recv2,pj->recv2_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  }
  if(pj->n_send2) {
    ierr = MPI_Waitall(pj->n_send2,pj->send2_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    for (i=0;i<pj->n_send2;i++) {
      ierr = MatDenseRestoreArray(submat[i],&array[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&submat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(submat,array);CHKERRQ(ierr);
  }

  /** perfoming the actual multiplication G_{rankG}^T*F*G_{neigh_rankG>=rankG} */   
  if (ft->n_cs) {
    for (i=0;i<pj->n_sum_mats;i++) { ierr = MatZeroEntries(pj->sum_mats[i]);CHKERRQ(ierr); }
    ierr = PetscMalloc1(ft->n_lambda_local,&idxm);CHKERRQ(ierr);
    for (i=0;i<ft->n_lambda_local;i++) idxm[i] = i;
    /* sum F*G_{neigh_rankG>=rankG} */
    for (idx=0,delta=0,i=0;i<ft->n_neigh_lb;i++) {
      k0    = ft->neigh_lb[i];
      n_cs = pj->n_cs_comm[k0];
      if (n_cs) {
	if (k0>=rankG) {	
	  ierr = PetscFindInt(k0,pj->n_sum_mats,pj->i2rank,&jdx);CHKERRQ(ierr);
	  ierr = MatDenseGetArray(pj->FGholder[idx],&m_pointer);CHKERRQ(ierr);
	  ierr = MatSetValuesBlocked(pj->sum_mats[jdx],ft->n_lambda_local,idxm,n_cs,idxn,m_pointer,ADD_VALUES);CHKERRQ(ierr);
	  ierr = MatDenseRestoreArray(pj->FGholder[idx],&m_pointer);CHKERRQ(ierr);
	}
	idx++;
      }
      if (i) {
	for (j=0;j<pj->n_neighs2[i-1];j++) { 
	  k     = pj->neighs2[i-1][j];
	  n_cs = pj->n_cs_comm[k];
	  if(rankG<=k && n_cs) {
	    ierr = PetscFindInt(k,pj->n_sum_mats,pj->i2rank,&jdx);CHKERRQ(ierr);
	    ierr = MatSetValuesBlocked(pj->sum_mats[jdx],ft->n_shared_lb[i],ft->shared_lb[i],n_cs,idxn,&pj->fgmatrices[delta],ADD_VALUES);CHKERRQ(ierr);
	    delta += n_cs*ft->n_shared_lb[i]; 
	  }
	}
      }
    }
    for (i=0;i<pj->n_sum_mats;i++) { ierr = MatAssemblyBegin(pj->sum_mats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); }
    for (i=0;i<pj->n_sum_mats;i++) { ierr = MatAssemblyEnd  (pj->sum_mats[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); }

    /* multiply by G^T */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,pj->localnnz,pj->bufferPSum,&aux_mat);CHKERRQ(ierr);
    ierr = MatSetOption(aux_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(ft->localG,aux_mat,MAT_REUSE_MATRIX,PETSC_DEFAULT,&pj->local_rows_matrix);CHKERRQ(ierr);
    ierr = MatDestroy(&aux_mat);CHKERRQ(ierr);   
    ierr = PetscFree(idxm);CHKERRQ(ierr);
    ierr = PetscFree(idxn);CHKERRQ(ierr);
    
    /** local "row block" contribution to G^T*G */
    ierr = MatDenseGetArray(pj->local_rows_matrix,&m_pointer);CHKERRQ(ierr);
  }
  
  /* assemble matrix */
  /* gather values for the coarse problem's matrix and assemble it */ 
  for (k=0,k0=0,i=0; i<sizeG; i++) {
    n_cs = pj->n_cs_comm[i];
    if(n_cs>0){
      ierr = PetscMPIIntCast(n_cs*pj->c_count[i],&i_mpi0);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(i,&i_mpi1);CHKERRQ(ierr);
      if(i==rankG) {
	ierr = MPI_Bcast(m_pointer,i_mpi0,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(pj->coarse_problem,n_cs,&pj->r_coarse[k],pj->c_count[i],&pj->c_coarse[k0],m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(pj->local_rows_matrix,&m_pointer);CHKERRQ(ierr);
      } else {
	ierr = PetscMalloc1(i_mpi0,&m_pointer1);CHKERRQ(ierr);	  
	ierr = MPI_Bcast(m_pointer1,i_mpi0,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(pj->coarse_problem,n_cs,&pj->r_coarse[k],pj->c_count[i],&pj->c_coarse[k0],m_pointer1,INSERT_VALUES);CHKERRQ(ierr);
	ierr = PetscFree(m_pointer1);CHKERRQ(ierr);
      }
      k0 += pj->c_count[i];
      k  += n_cs;
    }
  }
  ierr = MatAssemblyBegin(pj->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pj->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJFactorizeCoarseProblem_PJ2LEVEL"
static PetscErrorCode FETIPJFactorizeCoarseProblem_PJ2LEVEL(FETIPJ ftpj)
{
  PetscErrorCode    ierr;
  FETI              ft  = ftpj->feti;
  PJ2LEVEL          *pj = (PJ2LEVEL*)ftpj->data;
  PC                pc;
  
  PetscFunctionBegin;
  if(!pj->coarse_problem) SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: FETI2SetUpCoarseProblem_Private() must be first called");
  
  /* factorize the coarse problem */
  if(!pj->ksp_coarse) {
    ierr = KSPCreate(PETSC_COMM_SELF,&pj->ksp_coarse);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pj->ksp_coarse,(PetscObject)pj,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pj,(PetscObject)pj->ksp_coarse);CHKERRQ(ierr);
    ierr = KSPSetType(pj->ksp_coarse,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pj->ksp_coarse,"feti_pj2level_pc_coarse_");CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(pj->coarse_problem,"feti_pj2level_pc_coarse_");CHKERRQ(ierr);
    ierr = KSPGetPC(pj->ksp_coarse,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(pj->ksp_coarse);CHKERRQ(ierr);
    ierr = KSPSetOperators(pj->ksp_coarse,pj->coarse_problem,pj->coarse_problem);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
  } else {
    ierr = KSPGetPC(pj->ksp_coarse,&pc);CHKERRQ(ierr);
    ierr = KSPSetOperators(pj->ksp_coarse,pj->coarse_problem,pj->coarse_problem);CHKERRQ(ierr);
    ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
  
  }
  ierr = KSPSetUp(pj->ksp_coarse);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&pj->F_coarse);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJSetUp_PJ2LEVEL"
static PetscErrorCode FETIPJSetUp_PJ2LEVEL(FETIPJ ftpj)
{
  PetscErrorCode    ierr;
  FETI              ft  = ftpj->feti;
  PJ2LEVEL          *pj = (PJ2LEVEL*)ftpj->data;
  Subdomain         sd = ft->subdomain;
  PetscMPIInt       i_mpi,sizeG,*c_displ,rankG,n_send,n_recv;
  PetscInt          k,k0,total_c_coarse,*idxm=NULL,*idxn=NULL,*idxa=NULL;
  /* nnz: array containing the number of block nonzeros in the upper triangular plus diagonal portion of each block*/
  PetscInt          i,j,idx,*nnz=NULL,size_floating,total_size_matrices=0;
  MPI_Comm          comm;
  PetscInt          *local_neighs=NULL,n_local_neighs,total_sz_fgmatrices,jdx,kdx,n_cs;

  PetscFunctionBegin;
  /* set projections in ksp */
  ierr = KSPSetProjection(ft->ksp_interface,FETIPJProject_PJ2LEVEL,(void*)ft);CHKERRQ(ierr);
  ierr = KSPSetReProjection(ft->ksp_interface,FETIPJReProject_PJ2LEVEL,(void*)ft);CHKERRQ(ierr);

  /* In the following structures for assembling and computing the coarse problem are setup */
  /* rankG and sizeG are related to the MPI_Comm comm */
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rankG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&sizeG);CHKERRQ(ierr);

  /* ====>>> Computing information of neighbours of neighbours */
  ierr  = PetscMalloc1(ft->n_neigh_lb-1,&pj->n_neighs2);CHKERRQ(ierr);  /* n_neighs2[0] != ft->n_neigh_lb; not count myself */
  ierr  = PetscMalloc1(ft->n_neigh_lb-1,&pj->send_reqs);CHKERRQ(ierr);
  ierr  = PetscMalloc1(ft->n_neigh_lb-1,&pj->recv_reqs);CHKERRQ(ierr);
  /* n_neighs2[0] != ft->n_neigh_lb; not count myself */
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Isend(&ft->n_neigh_lb,1,MPIU_INT,i_mpi,0,comm,&pj->send_reqs[i-1]);CHKERRQ(ierr);
    ierr = MPI_Irecv(&pj->n_neighs2[i-1],1,MPIU_INT,i_mpi,0,comm,&pj->recv_reqs[i-1]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(ft->n_neigh_lb-1,pj->recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(ft->n_neigh_lb-1,pj->send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

  n_local_neighs = 0;
  for (i=1; i<ft->n_neigh_lb; i++){
    n_local_neighs  += pj->n_neighs2[i-1];
  }
  
  ierr = PetscMalloc1(ft->n_neigh_lb-1,&pj->neighs2);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_local_neighs,&pj->neighs2[0]);CHKERRQ(ierr);
  for (i=1;i<ft->n_neigh_lb-1;i++) {
    pj->neighs2[i] = pj->neighs2[i-1] + pj->n_neighs2[i-1];
  }
  for (i=1; i<ft->n_neigh_lb; i++){
    ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
    ierr = MPI_Isend(ft->neigh_lb,ft->n_neigh_lb,MPIU_INT,i_mpi,0,comm,&pj->send_reqs[i-1]);CHKERRQ(ierr);
    ierr = MPI_Irecv(pj->neighs2[i-1],pj->n_neighs2[i-1],MPIU_INT,i_mpi,0,comm,&pj->recv_reqs[i-1]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(ft->n_neigh_lb-1,pj->recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(ft->n_neigh_lb-1,pj->send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree(pj->send_reqs);CHKERRQ(ierr);
  ierr = PetscFree(pj->recv_reqs);CHKERRQ(ierr);
  /* ====<<< Computing information of neighbours of neighbours */

  
  /* computing n_cs_comm that is number of rbm per subdomain and the communicator of floating structures */
  ierr = PetscMalloc1(sizeG,&pj->n_cs_comm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&ft->n_cs,1,MPIU_INT,pj->n_cs_comm,1,MPIU_INT,comm);CHKERRQ(ierr);
  
  /* computing displ structure for next allgatherv and total number of rbm of the system */
  ierr               = PetscMalloc1(sizeG,&pj->displ);CHKERRQ(ierr);
  pj->displ[0]      = 0;
  n_cs              = pj->n_cs_comm[0];
  pj->total_rbm     = n_cs;
  size_floating      = (n_cs>0);
  for (i=1;i<sizeG;i++){
    n_cs              = pj->n_cs_comm[i];
    pj->total_rbm    += n_cs;
    size_floating     += (n_cs>0);
    pj->displ[i]      = pj->displ[i-1] + pj->n_cs_comm[i-1];
  }

  /* localnnz: nonzeros for my row of the coarse probem */
  if(ft->n_cs>0) {ierr = PetscMalloc1(n_local_neighs,&local_neighs);CHKERRQ(ierr);}
  n_local_neighs      = 0;
  pj->localnnz       = 0;
  total_size_matrices = 0;
  total_sz_fgmatrices = 0;
  pj->max_n_cs      = ft->n_cs;
  n_send              = (ft->n_neigh_lb-1)*(ft->n_cs>0);
  n_recv              = 0;
  pj->n_send2        = 0;
  pj->n_recv2        = 0;
  for (i=1;i<ft->n_neigh_lb;i++) {
    k0      = ft->neigh_lb[i];
    n_cs   = pj->n_cs_comm[k0];
    /* for communicating Gs */
    n_recv              += (n_cs>0);
    total_size_matrices += n_cs*ft->n_shared_lb[i];
    /* I send */
    pj->n_send2  += (rankG>k0)*(n_cs>0)*(ft->n_cs>0);/*send myself Fs*Gs*/
    for (j=1;j<ft->n_neigh_lb;j++) { /*send Fs*G_{my_neighs}U{k0}*/
      k      = ft->neigh_lb[j];
      n_cs  = pj->n_cs_comm[k];
      if(k>=k0) {
	pj->n_send2  += (n_cs>0);
      }
    }  
    /* I receive */
    for (j=0;j<pj->n_neighs2[i-1];j++) { /*receive F_{my_neighs}*G_{my_neighs}U{neighs_neighs}*/
      k              = pj->neighs2[i-1][j];
      n_cs          = pj->n_cs_comm[k];
      pj->max_n_cs = (pj->max_n_cs > n_cs) ? pj->max_n_cs : n_cs;
      if(rankG<=k && n_cs>0 && ft->n_cs>0) {
	pj->n_recv2++;
	total_sz_fgmatrices += n_cs*ft->n_shared_lb[i];
	local_neighs[n_local_neighs++] = k;
      }
    }
  }
  
  pj->n_sum_mats = 0;
  if (ft->n_cs>0) {
    ierr            = PetscSortRemoveDupsInt(&n_local_neighs,local_neighs);CHKERRQ(ierr);
    for (i=0;i<n_local_neighs;i++) {
      n_cs            = pj->n_cs_comm[local_neighs[i]];
      pj->localnnz   += n_cs;
      pj->n_sum_mats ++; /* += (n_cs>0); */
    }
    ierr = PetscMalloc1(ft->n_lambda_local*pj->localnnz,&pj->bufferPSum);CHKERRQ(ierr);
    ierr = PetscMalloc2(pj->n_sum_mats,&pj->sum_mats,pj->n_sum_mats,&pj->i2rank);CHKERRQ(ierr);
    ierr = PetscMalloc1(pj->localnnz,&idxn);CHKERRQ(ierr);
    ierr = PetscMalloc1(ft->n_cs,&idxm);CHKERRQ(ierr);
    /** row indices */
    jdx  = pj->displ[rankG];
    for (i=0; i<ft->n_cs; i++) idxm[i] = i + jdx;

    for (kdx=0,k=0,i=0;i<n_local_neighs;i++) {
      k0             = local_neighs[i];
      n_cs          = pj->n_cs_comm[k0];
      pj->i2rank[i] = k0;
      ierr           = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,n_cs,pj->bufferPSum+kdx,&pj->sum_mats[i]);CHKERRQ(ierr);
      ierr           = MatSetOption(pj->sum_mats[i],MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
      kdx           += ft->n_lambda_local*n_cs;
      /** col indices */
      jdx = pj->displ[k0];
      for (j=0;j<n_cs;j++, k++) idxn[k] = j + jdx;
    }    
    ierr = PetscFree(local_neighs);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_cs,pj->localnnz,NULL,&pj->local_rows_matrix);CHKERRQ(ierr);
    ierr = MatSetOption(pj->local_rows_matrix,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  }
  
  ierr = PetscMalloc1(pj->total_rbm,&nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(size_floating,&idxa);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_displ);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&pj->c_count);CHKERRQ(ierr);
  pj->c_count[0] = (pj->n_cs_comm[0]>0);
  c_displ[0]      = 0;
  for (i=1;i<sizeG;i++) {
    c_displ[i]      = c_displ[i-1] + pj->c_count[i-1];
    pj->c_count[i] = (pj->n_cs_comm[i]>0);
  }
  ierr = MPI_Allgatherv(&pj->localnnz,(pj->n_cs_comm[rankG]>0),MPIU_INT,idxa,pj->c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);
  for (k0=0,k=0,i=0;i<sizeG;i++) {
    n_cs = pj->n_cs_comm[i];
    for(j=0;j<n_cs;j++) nnz[k++] = idxa[k0];
    k0 += (n_cs>0);
  }
  ierr = PetscFree(idxa);CHKERRQ(ierr);

  /* create the "global" matrix for holding G^T*F*G */
  ierr = MatDestroy(&pj->coarse_problem);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&pj->coarse_problem);CHKERRQ(ierr);
  ierr = MatSetType(pj->coarse_problem,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(pj->coarse_problem,1);CHKERRQ(ierr);
  ierr = MatSetSizes(pj->coarse_problem,pj->total_rbm,pj->total_rbm,pj->total_rbm,pj->total_rbm);CHKERRQ(ierr);
  ierr = MatSetOption(pj->coarse_problem,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(pj->coarse_problem,1,PETSC_DEFAULT,nnz);CHKERRQ(ierr);
  ierr = MatSetUp(pj->coarse_problem);CHKERRQ(ierr);

  /* Structures for communicating matrices G with my neighbors */
  pj->n_send = n_send;
  pj->n_recv = n_recv;
  if(n_send) { ierr = PetscMalloc1(n_send,&pj->send_reqs);CHKERRQ(ierr);}
  if(n_recv) {
    ierr = PetscMalloc1(total_size_matrices,&pj->matrices);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_recv,&pj->recv_reqs);CHKERRQ(ierr);
  }
  if(n_recv) {
    /* store received matrices in Gholder */
    pj->n_Gholder = n_recv;
    ierr = PetscMalloc1(n_recv,&pj->Gholder);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_recv,&pj->neigh_holder);CHKERRQ(ierr);
    ierr = PetscMalloc1(2*n_recv,&pj->neigh_holder[0]);CHKERRQ(ierr);
    for (i=1;i<n_recv;i++) { 
      pj->neigh_holder[i] = pj->neigh_holder[i-1] + 2;
    }
    for (i=0,idx=0,k=1; k<ft->n_neigh_lb; k++){
      if (pj->n_cs_comm[ft->neigh_lb[k]]>0) {
	pj->neigh_holder[i][0] = ft->neigh_lb[k];
	pj->neigh_holder[i][1] = k;
	ierr  = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_shared_lb[k],pj->n_cs_comm[ft->neigh_lb[k]],&pj->matrices[idx],&pj->Gholder[i]);CHKERRQ(ierr);
	ierr  = MatSetOption(pj->Gholder[i++],MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
	idx  += pj->n_cs_comm[ft->neigh_lb[k]]*ft->n_shared_lb[k];
      }
    }
  }

  /* creating strucutres for computing F_local*G_neighbors */
  ierr = PetscMalloc3(sd->n*pj->max_n_cs,&pj->bufferRHS,sd->n*pj->max_n_cs,&pj->bufferX,ft->n_lambda_local*pj->max_n_cs,&pj->bufferG);CHKERRQ(ierr);

  pj->n_FGholder = n_recv + (ft->n_cs>0);
  ierr            = PetscMalloc1(pj->n_FGholder,&pj->FGholder);CHKERRQ(ierr);
  for (i=0,k=0; k<ft->n_neigh_lb; k++) {
    n_cs = pj->n_cs_comm[ft->neigh_lb[k]];
    if (n_cs)  {
      /* the following matrix is created using in column major order (the usual Fortran 77 manner) */
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,n_cs,NULL,&pj->FGholder[i]);CHKERRQ(ierr);
      ierr = MatSetOption(pj->FGholder[i++],MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  /* creating strucutres for communicating computed FG */
  if(pj->n_send2) { ierr = PetscMalloc1(pj->n_send2,&pj->send2_reqs);CHKERRQ(ierr);}
  if(pj->n_recv2) {
    ierr = PetscMalloc1(total_sz_fgmatrices,&pj->fgmatrices);CHKERRQ(ierr);
    ierr = PetscMalloc1(pj->n_recv2,&pj->recv2_reqs);CHKERRQ(ierr);
  } 

  /* creating structures for assembling the matrix for the coarse problem */
  ierr            = PetscMalloc1(pj->total_rbm,&pj->r_coarse);CHKERRQ(ierr);
  n_cs           = pj->n_cs_comm[0];
  total_c_coarse  = nnz[0]*(n_cs>0);
  pj->c_count[0] = nnz[0]*(n_cs>0);
  c_displ[0]      = 0;
  idx             = n_cs;
  for (i=1;i<sizeG;i++) {
    n_cs             = pj->n_cs_comm[i];
    total_c_coarse   += nnz[idx]*(n_cs>0);
    c_displ[i]        = c_displ[i-1] + pj->c_count[i-1];
    pj->c_count[i]   = nnz[idx]*(n_cs>0);
    idx              += n_cs;
  }
  ierr = PetscMalloc1(total_c_coarse,&pj->c_coarse);CHKERRQ(ierr);
  /* gather rows and columns*/
  ierr = MPI_Allgatherv(idxm,ft->n_cs,MPIU_INT,pj->r_coarse,pj->n_cs_comm,pj->displ,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(idxn,pj->localnnz,MPIU_INT,pj->c_coarse,pj->c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);

  ierr = PetscFree(idxm);CHKERRQ(ierr);
  ierr = PetscFree(idxn);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscFree(c_displ);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJDestroy_PJ2LEVEL_CS"
static PetscErrorCode FETIPJDestroy_PJ2LEVEL_CS(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  PJ2LEVEL       *pj = (PJ2LEVEL*)ftpj->data;
  PetscInt       i;
   
  PetscFunctionBegin;  
  ierr = PetscFree3(pj->bufferRHS,pj->bufferX,pj->bufferG);CHKERRQ(ierr);
  ierr = PetscFree(pj->recv2_reqs);CHKERRQ(ierr);
  ierr = PetscFree(pj->fgmatrices);CHKERRQ(ierr);
  ierr = PetscFree(pj->send2_reqs);CHKERRQ(ierr);
  ierr = MatDestroy(&pj->local_rows_matrix);CHKERRQ(ierr);  
  for (i=0;i<pj->n_sum_mats;i++) {
    ierr = MatDestroy(&pj->sum_mats[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(pj->sum_mats,pj->i2rank);CHKERRQ(ierr);
  ierr = PetscFree(pj->bufferPSum);CHKERRQ(ierr);
  ierr = PetscFree(pj->c_coarse);CHKERRQ(ierr);
  ierr = PetscFree(pj->r_coarse);CHKERRQ(ierr);
  ierr = PetscFree(pj->c_count);CHKERRQ(ierr);
  ierr = PetscFree(pj->n_cs_comm);CHKERRQ(ierr);
  ierr = PetscFree(pj->n_neighs2);CHKERRQ(ierr);
  ierr = PetscFree(pj->neighs2[0]);CHKERRQ(ierr);
  ierr = PetscFree(pj->neighs2);CHKERRQ(ierr);
  if (pj->FGholder) {
    for (i=0;i<pj->n_FGholder;i++) { ierr = MatDestroy(&pj->FGholder[i]);CHKERRQ(ierr); }
    ierr = PetscFree(pj->FGholder);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJDestroy_PJ2LEVEL_GATHER_NEIGH"
static PetscErrorCode FETIPJDestroy_PJ2LEVEL_GATHER_NEIGH(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  PJ2LEVEL       *pj = (PJ2LEVEL*)ftpj->data;
   
  PetscFunctionBegin;  
  if(pj->n_recv) { ierr = PetscFree(pj->recv_reqs);CHKERRQ(ierr);}
  if(pj->n_send) { ierr = PetscFree(pj->send_reqs);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJDestroy_PJ2LEVEL"
static PetscErrorCode FETIPJDestroy_PJ2LEVEL(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  PJ2LEVEL       *pj = (PJ2LEVEL*)ftpj->data;
  PetscInt       i;
  
  PetscFunctionBegin;
  ierr = FETIPJDestroy_PJ2LEVEL_GATHER_NEIGH(ftpj);CHKERRQ(ierr);
  ierr = FETIPJDestroy_PJ2LEVEL_CS(ftpj);CHKERRQ(ierr);

  ierr = KSPDestroy(&pj->ksp_coarse);CHKERRQ(ierr);
  if(pj->neigh_holder) {
    ierr = PetscFree(pj->neigh_holder[0]);CHKERRQ(ierr);
    ierr = PetscFree(pj->neigh_holder);CHKERRQ(ierr);
  }
  if(pj->displ) { ierr = PetscFree(pj->displ);CHKERRQ(ierr);}
  for (i=0;i<pj->n_Gholder;i++) {
    ierr = MatDestroy(&pj->Gholder[i]);CHKERRQ(ierr);
  }
  if(pj->coarse_problem) {ierr = MatDestroy(&pj->coarse_problem);CHKERRQ(ierr);}
  ierr = PetscFree(pj->Gholder);CHKERRQ(ierr);
  if(pj->matrices) { ierr = PetscFree(pj->matrices);CHKERRQ(ierr);}

  ierr = PetscFree(ftpj->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJApplyCoarseProblem_PJ2LEVEL"
/*@
   FETIPJApplyCoarseProblem_PJ2LEVEL - Applies the operation G*(G^T*F*G)^{-1} to a vector  

   Input Parameter:
.  ftpj - the FETIPJ context
.  v    - the input vector (in this case v is a local copy of the global assembled vector)

   Output Parameter:
.  r  - the output vector. 

   Level: developer

.keywords: FETI projection

@*/
static PetscErrorCode FETIPJApplyCoarseProblem_PJ2LEVEL(FETIPJ ftpj,Vec v,Vec r)
{
  PetscErrorCode     ierr;
  FETI               ft = ftpj->feti;
  PJ2LEVEL           *pj = (PJ2LEVEL*)ftpj->data;
  Vec                v_rbm; /* vec of dimension total_rbm */
  Vec                v0;    /* vec of dimension n_cs */
  Vec                r_local,vec_holder;
  IS                 subset;
  PetscMPIInt        rank;
  MPI_Comm           comm;
  PetscInt           i,j,idx0,idx1,*indices;
  const PetscScalar  *m_pointer; 
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  if(!pj->F_coarse) SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Error: FETIPJFactorizeCoarseProblem() must be first called");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  
  /* apply (G^T*G)^{-1}: compute v_rbm = (G^T*G)^{-1}*v */
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&v_rbm);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(pj->F_coarse,25,0);CHKERRQ(ierr);
#endif
  ierr = MatSolve(pj->F_coarse,v,v_rbm);CHKERRQ(ierr);

  /* apply G: compute r = G*v_rbm */
  ierr = VecUnAsmGetLocalVector(r,&r_local);CHKERRQ(ierr);
  ierr = PetscMalloc1(pj->max_n_cs,&indices);CHKERRQ(ierr);
  /** mulplying by localG for the current processor */
  if(ft->n_cs) {
    for (i=0;i<ft->n_cs;i++) indices[i] = pj->displ[rank] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_cs,indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
    ierr = VecGetSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = MatMult(ft->localG,v0,r_local);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = ISDestroy(&subset);CHKERRQ(ierr);
  } else {
    ierr = VecSet(r_local,0.0);CHKERRQ(ierr);
  }
  /** multiplying by localG for the other processors */
  for (j=0;j<pj->n_Gholder;j++) {
    idx0 = pj->neigh_holder[j][0];
    idx1 = pj->neigh_holder[j][1];   
    for (i=0;i<pj->n_cs_comm[idx0];i++) indices[i] = pj->displ[idx0] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,pj->n_cs_comm[idx0],indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
    ierr = VecGetSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_shared_lb[idx1],&vec_holder);CHKERRQ(ierr);
    ierr = MatMult(pj->Gholder[j],v0,vec_holder);CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec_holder,&m_pointer);CHKERRQ(ierr);
    ierr = VecSetValues(r_local,ft->n_shared_lb[idx1],ft->shared_lb[idx1],m_pointer,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(v_rbm,subset,&v0);CHKERRQ(ierr);
    ierr = ISDestroy(&subset);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(vec_holder,&m_pointer);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_holder);CHKERRQ(ierr); 
  }
  
  ierr = VecAssemblyBegin(r_local);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(r_local);CHKERRQ(ierr);

  ierr = VecUnAsmRestoreLocalVector(r,r_local);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = VecDestroy(&v_rbm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJProject_PJ2LEVEL"
/*@
   FETIPJProject_PJ2LEVEL - Performs the projection step of FETIPJ 2 level.

   Input Parameter:
.  ft        - the FETI context
.  g_global  - the vector to project
.  y         - the projected vector

   Level: developer

.keywords: FETI projection two level

@*/
PetscErrorCode FETIPJProject_PJ2LEVEL(void* ft_ctx, Vec g_global, Vec y)
{
  PetscErrorCode    ierr;
  FETIPJ            ftpj; 
  FETI              ft;
  PJ2LEVEL          *pj;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  Vec               lambda_local=0,localv=0,y_aux=0;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft_ctx,FETI_CLASSID,1);
  PetscValidHeaderSpecific(g_global,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ft   = (FETI)ft_ctx;
  ftpj = ft->ftpj;
  pj   = (PJ2LEVEL*)ftpj->data;
  comm = PetscObjectComm((PetscObject)ft);
  if (y == g_global) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use g_global == y");
  ierr = VecUnAsmGetLocalVectorRead(g_global,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,pj->n_cs_comm,pj->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(g_global,lambda_local);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETIPJApplyCoarseProblem_PJ2LEVEL(ftpj,asm_e,y);CHKERRQ(ierr);

  ierr = VecDuplicate(y,&y_aux);CHKERRQ(ierr);  
  ierr = MatMultFlambda_FETI(ft,y,y_aux);CHKERRQ(ierr);

  ierr = VecWAXPY(y,-1,y_aux,g_global);CHKERRQ(ierr);

  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  ierr = VecDestroy(&y_aux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJReProject_PJ2LEVEL"
/*@
   FETIPJReProject_PJ2LEVEL - Performs the re-projection step of FETIPJ 2 level.

   Input Parameter:
.  ft        - the FETI context
.  g_global  - the vector to project
.  y         - the projected vector

   Level: developer

.keywords: FETI projection two level

@*/
PetscErrorCode FETIPJReProject_PJ2LEVEL(void* ft_ctx, Vec g_global, Vec y)
{
  PetscErrorCode    ierr;
  FETIPJ            ftpj; 
  FETI              ft;
  PJ2LEVEL          *pj;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  Vec               lambda_local,localv;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ft_ctx,FETI_CLASSID,1);
  PetscValidHeaderSpecific(g_global,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ft   = (FETI)ft_ctx;
  ftpj = ft->ftpj;
  pj   = (PJ2LEVEL*)ftpj->data;
  comm = PetscObjectComm((PetscObject)ft);
  if (y == g_global) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use g_global == y");
  
  ierr = MatMultFlambda_FETI(ft,g_global,y);CHKERRQ(ierr);

  ierr = VecUnAsmGetLocalVectorRead(y,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,pj->n_cs_comm,pj->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(y,lambda_local);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETIPJApplyCoarseProblem_PJ2LEVEL(ftpj,asm_e,y);CHKERRQ(ierr);
  ierr = VecAYPX(y,-1,g_global);CHKERRQ(ierr);

  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJComputeInitialCondition_PJ2LEVEL"
static PetscErrorCode FETIPJComputeInitialCondition_PJ2LEVEL(FETIPJ ftpj)
{
  PetscErrorCode    ierr;
  FETI              ft = ftpj->feti;
  PJ2LEVEL          *pj = (PJ2LEVEL*)ftpj->data;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  Vec               lambda_local,localv;
  
  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)ft);
  ierr = VecUnAsmGetLocalVectorRead(ft->d,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecUnAsmRestoreLocalVectorRead(ft->d,lambda_local);CHKERRQ(ierr);
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,pj->n_cs_comm,pj->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETIPJApplyCoarseProblem_PJ2LEVEL(ftpj,asm_e,ft->lambda_global);CHKERRQ(ierr);

  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJCreate_PJ2LEVEL"
PETSC_EXTERN PetscErrorCode FETIPJCreate_PJ2LEVEL(FETIPJ);
PetscErrorCode FETIPJCreate_PJ2LEVEL(FETIPJ ftpj)
{
  PetscErrorCode  ierr;
  FETI            ft = ftpj->feti;
  PJ2LEVEL        *pj = (PJ2LEVEL*)ftpj->data;
  
  PetscFunctionBegin;
  ierr       = PetscNewLog(ft,&pj);CHKERRQ(ierr);
  ftpj->data = (void*)pj;

  ierr  = PetscMemzero(pj,sizeof(PJ2LEVEL));CHKERRQ(ierr);

  ftpj->ops->setup               = FETIPJSetUp_PJ2LEVEL;
  ftpj->ops->destroy             = FETIPJDestroy_PJ2LEVEL;
  ftpj->ops->setfromoptions      = 0;
  ftpj->ops->gatherneighbors     = FETIPJGatherNeighborsCoarseBasis_PJ2LEVEL;
  ftpj->ops->assemble            = FETIPJAssembleCoarseProblem_PJ2LEVEL;
  ftpj->ops->factorize           = FETIPJFactorizeCoarseProblem_PJ2LEVEL;
  ftpj->ops->initialcondition    = FETIPJComputeInitialCondition_PJ2LEVEL;
  PetscFunctionReturn(0);
}

