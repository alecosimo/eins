#include <../src/feti/einspj1level.h>
#include <private/einsvecimpl.h>
#include <petsc/private/matimpl.h>
#include <einsksp.h>
#include <einssys.h>
#include <einspc.h>


PetscErrorCode FETIPJProject_PJ1LEVEL(void*,Vec,Vec);


/* #undef __FUNCT__ */
/* #define __FUNCT__ "FETIPJGatherNeighborsCoarseBasis_PJ1LEVEL" */
/* static PetscErrorCode FETIPJGatherNeighborsCoarseBasis_PJ1LEVEL(FETIPJ ftpj) */
/* { */
/*   PetscErrorCode    ierr; */
/*   FETI              ft  = ftpj->feti; */
/*   PJ1LEVEL          *pj = (PJ1LEVEL*)ftpj->data; */
  
/*   PetscFunctionBegin; */
/*   PetscFunctionReturn(0); */
/* } */


/* #undef __FUNCT__ */
/* #define __FUNCT__ "FETIPJAssembleCoarseProblem_PJ1LEVEL" */
/* static PetscErrorCode FETIPJAssembleCoarseProblem_PJ1LEVEL(FETIPJ ftpj) */
/* { */
/*   PetscErrorCode    ierr; */
/*   FETI              ft  = ftpj->feti; */
/*   PJ1LEVEL          *pj = (PJ1LEVEL*)ftpj->data; */
  
/*   PetscFunctionBegin; */
/*   PetscFunctionReturn(0); */
/* } */


#undef __FUNCT__
#define __FUNCT__ "FETIPJFactorizeCoarseProblem_PJ1LEVEL"
static PetscErrorCode FETIPJFactorizeCoarseProblem_PJ1LEVEL(FETIPJ ftpj)
{
  PetscErrorCode    ierr;
  FETI              ft  = ftpj->feti;
  PJ1LEVEL          *pj = (PJ1LEVEL*)ftpj->data;
  PC                pc;
  
  PetscFunctionBegin;
  if(!pj->coarse_problem) SETERRQ(PetscObjectComm((PetscObject)ft),PETSC_ERR_ARG_WRONGSTATE,"Error: FETIPJSetUp() must be first called");

  /* factorize the coarse problem */
  ierr = KSPCreate(PETSC_COMM_SELF,&pj->ksp_coarse);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)pj->ksp_coarse,(PetscObject)pj,1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pj,(PetscObject)pj->ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetType(pj->ksp_coarse,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(pj->ksp_coarse,"feti_pj1level_pc_coarse_");CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(pj->coarse_problem,"feti_pj1level_pc_coarse_");CHKERRQ(ierr);
  ierr = KSPGetPC(pj->ksp_coarse,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(pj->ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOperators(pj->ksp_coarse,pj->coarse_problem,pj->coarse_problem);CHKERRQ(ierr);
  ierr = PCFactorSetUpMatSolverPackage(pc);CHKERRQ(ierr);
  ierr = KSPSetUp(pj->ksp_coarse);CHKERRQ(ierr);
  ierr = PCFactorGetMatrix(pc,&pj->F_coarse);CHKERRQ(ierr);
  if(pj->destroy_coarse) {
    ierr = MatDestroy(&pj->coarse_problem);CHKERRQ(ierr);
    pj->coarse_problem = 0;    
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJSetUp_PJ1LEVEL"
static PetscErrorCode FETIPJSetUp_PJ1LEVEL(FETIPJ ftpj)
{
  PetscErrorCode    ierr;
  FETI              ft  = ftpj->feti;
  PJ1LEVEL          *pj = (PJ1LEVEL*)ftpj->data;
  PetscMPIInt       i_mpi,i_mpi1,sizeG,size,*c_displ,*c_count,n_recv,n_send,rankG;
  PetscInt          k,k0,*c_coarse,*r_coarse,total_c_coarse,*idxm=NULL,*idxn=NULL;
  /* nnz: array containing the number of block nonzeros in the upper triangular plus diagonal portion of each block*/
  PetscInt          i,j,idx,*n_cs_comm,*nnz,size_floating,total_size_matrices=0,localnnz=0;
  PetscScalar       *m_pointer=NULL,*m_pointer1=NULL,**array=NULL;
  MPI_Comm          comm;
  MPI_Request       *send_reqs=NULL,*recv_reqs=NULL;
  IS                isindex;
  /* Gholder: for holding non-local G that I receive from neighbours*/
  /* submat: submatrices of my own G to send to my neighbours */
  /* result: result of the local multiplication G^T*G*/
  Mat              *submat,result,aux_mat;

  PetscFunctionBegin;
  ierr = KSPSetProjection(ft->ksp_interface,FETIPJProject_PJ1LEVEL,(void*)ft);CHKERRQ(ierr);
  ierr = KSPSetReProjection(ft->ksp_interface,FETIPJProject_PJ1LEVEL,(void*)ft);CHKERRQ(ierr);

    /* in the following rankG and sizeG are related to the MPI_Comm comm */
  /* whereas rank and size are related to the MPI_Comm floatingComm */
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rankG);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&sizeG);CHKERRQ(ierr);
  
  /* computing n_cs_comm that is number of rbm per subdomain and the communicator of floating structures */
  ierr = PetscMalloc1(sizeG,&n_cs_comm);CHKERRQ(ierr);
  ierr = MPI_Allgather(&ft->n_cs,1,MPIU_INT,n_cs_comm,1,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Comm_split(comm,(n_cs_comm[rankG]>0),rankG,&pj->floatingComm);CHKERRQ(ierr);

  if (ft->n_cs){
    /* compute size and rank to the new communicator */
    ierr = MPI_Comm_size(pj->floatingComm,&size);CHKERRQ(ierr);
    /* computing displ_f and count_f_rbm */
    ierr                 = PetscMalloc1(size,&pj->displ_f);CHKERRQ(ierr);
    ierr                 = PetscMalloc1(size,&pj->count_f_rbm);CHKERRQ(ierr);
    pj->displ_f[0]      = 0;
    pj->count_f_rbm[0]  = n_cs_comm[0];
    k                    = (n_cs_comm[0]>0);
    for (i=1;i<sizeG;i++){
      if(n_cs_comm[i]) {
	pj->count_f_rbm[k] = n_cs_comm[i];
	pj->displ_f[k]     = pj->displ_f[k-1] + pj->count_f_rbm[k-1];
	k++;
      }
    }
  }
  
  /* computing displ structure for next allgatherv and total number of rbm of the system */
  ierr               = PetscMalloc1(sizeG,&pj->displ);CHKERRQ(ierr);
  ierr               = PetscMalloc1(sizeG,&pj->count_rbm);CHKERRQ(ierr);
  pj->displ[0]      = 0;
  pj->count_rbm[0]  = n_cs_comm[0];
  pj->total_rbm     = n_cs_comm[0];
  size_floating      = (n_cs_comm[0]>0);
  for (i=1;i<sizeG;i++){
    pj->total_rbm    += n_cs_comm[i];
    size_floating     += (n_cs_comm[i]>0);
    pj->count_rbm[i]  = n_cs_comm[i];
    pj->displ[i]      = pj->displ[i-1] + pj->count_rbm[i-1];
  }

  /* localnnz: nonzeros for my row of the coarse probem */
  localnnz            = ft->n_cs;
  total_size_matrices = 0;
  pj->max_n_cs      = ft->n_cs;
  n_send              = (ft->n_neigh_lb-1)*(ft->n_cs>0);
  n_recv              = 0;
  if(ft->n_cs) {
    for (i=1;i<ft->n_neigh_lb;i++){
      i_mpi                = n_cs_comm[ft->neigh_lb[i]];
      n_recv              += (i_mpi>0);
      pj->max_n_cs       = (pj->max_n_cs > i_mpi) ? pj->max_n_cs : i_mpi;
      total_size_matrices += i_mpi*ft->n_shared_lb[i];
      localnnz            += i_mpi*(ft->neigh_lb[i]>rankG);
    }
  } else {
    for (i=1;i<ft->n_neigh_lb;i++){
      i_mpi                = n_cs_comm[ft->neigh_lb[i]];
      n_recv              += (i_mpi>0);
      pj->max_n_cs       = (pj->max_n_cs > i_mpi) ? pj->max_n_cs : i_mpi;
      total_size_matrices += i_mpi*ft->n_shared_lb[i];
    }
  }

  ierr = PetscMalloc1(pj->total_rbm,&nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(size_floating,&idxm);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_displ);CHKERRQ(ierr);
  ierr = PetscMalloc1(sizeG,&c_count);CHKERRQ(ierr);
  c_count[0]     = (n_cs_comm[0]>0);
  c_displ[0]     = 0;
  for (i=1;i<sizeG;i++) {
    c_displ[i] = c_displ[i-1] + c_count[i-1];
    c_count[i] = (n_cs_comm[i]>0);
  }
  ierr = MPI_Allgatherv(&localnnz,(n_cs_comm[rankG]>0),MPIU_INT,idxm,c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);
  for (k0=0,k=0,i=0;i<sizeG;i++) {
    for(j=0;j<pj->count_rbm[i];j++) nnz[k++] = idxm[k0];
    k0 += (pj->count_rbm[i]>0);
  }
  ierr = PetscFree(idxm);CHKERRQ(ierr);

  /* create the "global" matrix for holding G^T*G */
  if(pj->destroy_coarse){ ierr = MatDestroy(&pj->coarse_problem);CHKERRQ(ierr);}
  ierr = MatCreate(PETSC_COMM_SELF,&pj->coarse_problem);CHKERRQ(ierr);
  ierr = MatSetType(pj->coarse_problem,MATSEQSBAIJ);CHKERRQ(ierr);
  ierr = MatSetBlockSize(pj->coarse_problem,1);CHKERRQ(ierr);
  ierr = MatSetSizes(pj->coarse_problem,pj->total_rbm,pj->total_rbm,pj->total_rbm,pj->total_rbm);CHKERRQ(ierr);
  ierr = MatSetOption(pj->coarse_problem,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(pj->coarse_problem,1,PETSC_DEFAULT,nnz);CHKERRQ(ierr);
  ierr = MatSetUp(pj->coarse_problem);CHKERRQ(ierr);

  /* Communicate matrices G */
  if(n_send) {
    ierr = PetscMalloc1(n_send,&send_reqs);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_send,&submat);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_send,&array);CHKERRQ(ierr);
    for (j=0, i=1; i<ft->n_neigh_lb; i++){
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ft->n_shared_lb[i],ft->shared_lb[i],PETSC_USE_POINTER,&isindex);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(ft->localG,isindex,NULL,MAT_INITIAL_MATRIX,&submat[j]);CHKERRQ(ierr);
      ierr = MatDenseGetArray(submat[j],&array[j]);CHKERRQ(ierr);   
      ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);   
      ierr = MPI_Isend(array[j],ft->n_cs*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,ft->tag,comm,&send_reqs[j]);CHKERRQ(ierr);
      ierr = ISDestroy(&isindex);CHKERRQ(ierr);
      j++;
    }
  }
  if(n_recv) {
    ierr = PetscMalloc1(total_size_matrices,&pj->matrices);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_recv,&recv_reqs);CHKERRQ(ierr);
    for (j=0,idx=0,i=1; i<ft->n_neigh_lb; i++){
      if (n_cs_comm[ft->neigh_lb[i]]>0) {
	ierr = PetscMPIIntCast(ft->neigh_lb[i],&i_mpi);CHKERRQ(ierr);
	ierr = MPI_Irecv(&pj->matrices[idx],n_cs_comm[ft->neigh_lb[i]]*ft->n_shared_lb[i],MPIU_SCALAR,i_mpi,ft->tag,comm,&recv_reqs[j]);CHKERRQ(ierr);    
	idx += n_cs_comm[ft->neigh_lb[i]]*ft->n_shared_lb[i]; 
	j++;
      }	  
    }
  }
  if(n_recv) { ierr = MPI_Waitall(n_recv,recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);}
  if(n_send) {
    ierr = MPI_Waitall(n_send,send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    for (i=0;i<n_send;i++) {
      ierr = MatDenseRestoreArray(submat[i],&array[i]);CHKERRQ(ierr);
      ierr = MatDestroy(&submat[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(array);CHKERRQ(ierr);
    ierr = PetscFree(submat);CHKERRQ(ierr);
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
      if (n_cs_comm[ft->neigh_lb[k]]>0) {
	pj->neigh_holder[i][0] = ft->neigh_lb[k];
	pj->neigh_holder[i][1] = k;
	ierr  = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_shared_lb[k],n_cs_comm[ft->neigh_lb[k]],&pj->matrices[idx],&pj->Gholder[i++]);CHKERRQ(ierr);
	idx  += n_cs_comm[ft->neigh_lb[k]]*ft->n_shared_lb[k];
      }
    }
  }
  
  /** perfoming the actual multiplication G_{rankG}^T*G_{neigh_rankG>=rankG} */   
  if (ft->n_cs) {
    ierr = PetscMalloc1(ft->n_lambda_local,&idxm);CHKERRQ(ierr);
    ierr = PetscMalloc1(pj->max_n_cs,&idxn);CHKERRQ(ierr);
    for (i=0;i<ft->n_lambda_local;i++) idxm[i] = i;
    for (i=0;i<ft->n_cs;i++) idxn[i] = i;
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_lambda_local,localnnz,NULL,&aux_mat);CHKERRQ(ierr);
    ierr = MatZeroEntries(aux_mat);CHKERRQ(ierr);
    ierr = MatSetOption(aux_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatDenseGetArray(ft->localG,&m_pointer);CHKERRQ(ierr);
    ierr = MatSetValuesBlocked(aux_mat,ft->n_lambda_local,idxm,ft->n_cs,idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(ft->localG,&m_pointer);CHKERRQ(ierr);

    for (k=0; k<pj->n_Gholder; k++){
      j   = pj->neigh_holder[k][0];
      idx = pj->neigh_holder[k][1];
      if (j>rankG) {
	for (k0=0;k0<n_cs_comm[j];k0++) idxn[k0] = i++;
	ierr = MatDenseGetArray(pj->Gholder[k],&m_pointer);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(aux_mat,ft->n_shared_lb[idx],ft->shared_lb[idx],n_cs_comm[j],idxn,m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(pj->Gholder[k],&m_pointer);CHKERRQ(ierr);	
      }
    }
    ierr = MatAssemblyBegin(aux_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(aux_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ft->n_cs,localnnz,NULL,&result);CHKERRQ(ierr);
    ierr = MatTransposeMatMult(ft->localG,aux_mat,MAT_REUSE_MATRIX,PETSC_DEFAULT,&result);CHKERRQ(ierr);
    ierr = MatDestroy(&aux_mat);CHKERRQ(ierr);
    ierr = PetscFree(idxm);CHKERRQ(ierr);
    ierr = PetscFree(idxn);CHKERRQ(ierr);
    
    /* building structures for assembling the "global matrix" of the coarse problem */
    ierr   = PetscMalloc1(ft->n_cs,&idxm);CHKERRQ(ierr);
    ierr   = PetscMalloc1(localnnz,&idxn);CHKERRQ(ierr);
    /** row indices */
    idx = pj->displ[rankG];
    for (i=0; i<ft->n_cs; i++) idxm[i] = i + idx;
    /** col indices */
    for (i=0; i<ft->n_cs; i++) idxn[i] = i + idx;
    for (j=1; j<ft->n_neigh_lb; j++) {
      k0  = n_cs_comm[ft->neigh_lb[j]];
      if ((ft->neigh_lb[j]>rankG)&&(k0>0)) {
	idx = pj->displ[ft->neigh_lb[j]];
	for (k=0;k<k0;k++, i++) idxn[i] = k + idx;
      }
    }
    /** local "row block" contribution to G^T*G */
    ierr = MatDenseGetArray(result,&m_pointer);CHKERRQ(ierr);
  }
  
  /* assemble matrix */
  ierr           = PetscMalloc1(pj->total_rbm,&r_coarse);CHKERRQ(ierr);
  total_c_coarse = nnz[0]*(n_cs_comm[0]>0);
  c_count[0]     = nnz[0]*(n_cs_comm[0]>0);
  c_displ[0]     = 0;
  idx            = n_cs_comm[0];
  for (i=1;i<sizeG;i++) {
    total_c_coarse += nnz[idx]*(n_cs_comm[i]>0);
    c_displ[i]      = c_displ[i-1] + c_count[i-1];
    c_count[i]      = nnz[idx]*(n_cs_comm[i]>0);
    idx            += n_cs_comm[i];
  }
  ierr = PetscMalloc1(total_c_coarse,&c_coarse);CHKERRQ(ierr);
  /* gather rows and columns*/
  ierr = MPI_Allgatherv(idxm,ft->n_cs,MPIU_INT,r_coarse,pj->count_rbm,pj->displ,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(idxn,localnnz,MPIU_INT,c_coarse,c_count,c_displ,MPIU_INT,comm);CHKERRQ(ierr);

  /* gather values for the coarse problem's matrix and assemble it */
  ierr = MatZeroEntries(pj->coarse_problem);CHKERRQ(ierr);
  for (k=0,k0=0,i=0; i<sizeG; i++) {
    if(n_cs_comm[i]>0){
      ierr = PetscMPIIntCast(n_cs_comm[i]*c_count[i],&i_mpi);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(i,&i_mpi1);CHKERRQ(ierr);
      if(i==rankG) {
	ierr = MPI_Bcast(m_pointer,i_mpi,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(pj->coarse_problem,n_cs_comm[i],&r_coarse[k],c_count[i],&c_coarse[k0],m_pointer,INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatDenseRestoreArray(result,&m_pointer);CHKERRQ(ierr);
	ierr = MatDestroy(&result);CHKERRQ(ierr);
      } else {
	ierr = PetscMalloc1(i_mpi,&m_pointer1);CHKERRQ(ierr);	  
	ierr = MPI_Bcast(m_pointer1,i_mpi,MPIU_SCALAR,i_mpi1,comm);CHKERRQ(ierr);
	ierr = MatSetValuesBlocked(pj->coarse_problem,n_cs_comm[i],&r_coarse[k],c_count[i],&c_coarse[k0],m_pointer1,INSERT_VALUES);CHKERRQ(ierr);
	ierr = PetscFree(m_pointer1);CHKERRQ(ierr);
      }
      k0 += c_count[i];
      k  += n_cs_comm[i];
    }
  }
  ierr = MatAssemblyBegin(pj->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pj->coarse_problem,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ftpj->state = FETIPJ_STATE_ASSEMBLED;
  
  ierr = PetscFree(idxm);CHKERRQ(ierr);
  ierr = PetscFree(idxn);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscFree(c_coarse);CHKERRQ(ierr);
  ierr = PetscFree(r_coarse);CHKERRQ(ierr);
  ierr = PetscFree(c_count);CHKERRQ(ierr);
  ierr = PetscFree(c_displ);CHKERRQ(ierr);
  ierr = PetscFree(send_reqs);CHKERRQ(ierr);
  ierr = PetscFree(recv_reqs);CHKERRQ(ierr);
  ierr = PetscFree(n_cs_comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJDestroy_PJ1LEVEL"
static PetscErrorCode FETIPJDestroy_PJ1LEVEL(FETIPJ ftpj)
{
  PetscErrorCode ierr;
  PJ1LEVEL       *pj = (PJ1LEVEL*)ftpj->data;
  PetscInt       i;
  
  PetscFunctionBegin;
  ierr = KSPDestroy(&pj->ksp_coarse);CHKERRQ(ierr);
  if(pj->neigh_holder) {
    ierr = PetscFree(pj->neigh_holder[0]);CHKERRQ(ierr);
    ierr = PetscFree(pj->neigh_holder);CHKERRQ(ierr);
  }
  if(pj->displ) { ierr = PetscFree(pj->displ);CHKERRQ(ierr);}
  if(pj->count_rbm) { ierr = PetscFree(pj->count_rbm);CHKERRQ(ierr);}
  if(pj->displ_f) { ierr = PetscFree(pj->displ_f);CHKERRQ(ierr);}
  if(pj->count_f_rbm) { ierr = PetscFree(pj->count_f_rbm);CHKERRQ(ierr);}
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
#define __FUNCT__ "FETIPJApplyCoarseProblem_PJ1LEVEL"
/*@
   FETIPJApplyCoarseProblem_PJ1LEVEL - Applies the operation G*(G^T*G)^{-1} to a vector  

   Input Parameter:
.  ftpj - the FETIPJ context
.  v    - the input vector (in this case v is a local copy of the global assembled vector)

   Output Parameter:
.  r  - the output vector. 

   Level: developer

.keywords: FETI projection

@*/
static PetscErrorCode FETIPJApplyCoarseProblem_PJ1LEVEL(FETIPJ ftpj,Vec v,Vec r)
{
  PetscErrorCode     ierr;
  FETI               ft = ftpj->feti;
  PJ1LEVEL           *pj = (PJ1LEVEL*)ftpj->data;
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
    for (i=0;i<pj->count_rbm[idx0];i++) indices[i] = pj->displ[idx0] + i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,pj->count_rbm[idx0],indices,PETSC_USE_POINTER,&subset);CHKERRQ(ierr);
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
#define __FUNCT__ "FETIPJProject_PJ1LEVEL"
/*@
   FETIPJProject_PJ1LEVEL - Performs the projection step of FETIPJ 1 level.

   Input Parameter:
.  ft        - the FETI context
.  g_global  - the vector to project
.  y         - the projected vector

   Level: developer

.keywords: FETI projection two level

@*/
PetscErrorCode FETIPJProject_PJ1LEVEL(void* ft_ctx, Vec g_global, Vec y)
{
  PetscErrorCode    ierr;
  FETIPJ            ftpj; 
  FETI              ft;
  PJ1LEVEL          *pj;
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
  pj   = (PJ1LEVEL*)ftpj->data;
    comm = PetscObjectComm((PetscObject)ft);
  if (y == g_global) SETERRQ(comm,PETSC_ERR_ARG_INCOMP,"Cannot use g_global == y");
  ierr = VecUnAsmGetLocalVectorRead(g_global,&lambda_local);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,ft->n_cs,&localv);CHKERRQ(ierr);
    ierr = MatMultTranspose(ft->localG,lambda_local,localv);CHKERRQ(ierr);   
    ierr = VecGetArrayRead(localv,&sbuff);CHKERRQ(ierr);
  }
  ierr = VecUnAsmRestoreLocalVectorRead(g_global,lambda_local);CHKERRQ(ierr);
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr); 
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,pj->count_rbm,pj->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft->n_cs) {
    ierr = VecRestoreArrayRead(localv,&sbuff);CHKERRQ(ierr);
    ierr = VecDestroy(&localv);CHKERRQ(ierr);
  }

  ierr = FETIPJApplyCoarseProblem_PJ1LEVEL(ftpj,asm_e,y);CHKERRQ(ierr);
  ierr = VecAYPX(y,-1,g_global);CHKERRQ(ierr);

  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJComputeAlphaNullSpace_PJ1LEVEL"
static PetscErrorCode FETIPJComputeAlphaNullSpace_PJ1LEVEL(FETIPJ ftpj,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  FETI              ft = ftpj->feti;
  PJ1LEVEL          *pj = (PJ1LEVEL*)ftpj->data;
  Vec               alpha_g,asm_g; 
  PetscMPIInt       rank;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  Vec               lambda_local;
  
  PetscFunctionBegin;
  if (!ft->n_cs)  PetscFunctionReturn(0);
  ierr = VecUnAsmGetLocalVectorRead(x,&lambda_local);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(pj->floatingComm,&rank);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&alpha_g);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&asm_g);CHKERRQ(ierr);  
  ierr = MatMultTranspose(ft->localG,lambda_local,y);CHKERRQ(ierr);   
  ierr = VecGetArrayRead(y,&sbuff);CHKERRQ(ierr);
  ierr = VecGetArray(asm_g,&rbuff);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,pj->count_f_rbm,pj->displ_f,MPIU_SCALAR,pj->floatingComm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_g,&rbuff);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(y,&sbuff);CHKERRQ(ierr);
  ierr = VecUnAsmRestoreLocalVectorRead(x,lambda_local);CHKERRQ(ierr);
  
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(pj->F_coarse,25,0);CHKERRQ(ierr);
#endif
  ierr = MatSolve(pj->F_coarse,asm_g,alpha_g);CHKERRQ(ierr);

  ierr = VecGetArrayRead(alpha_g,&sbuff);CHKERRQ(ierr);
  ierr = VecGetArray(y,&rbuff);CHKERRQ(ierr);
  ierr = PetscMemcpy(rbuff,sbuff+pj->displ_f[rank],sizeof(PetscScalar)*pj->count_f_rbm[rank]);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&rbuff);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(alpha_g,&sbuff);CHKERRQ(ierr);
  ierr = VecDestroy(&asm_g);CHKERRQ(ierr);
  ierr = VecDestroy(&alpha_g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FETIPJComputeInitialCondition_PJ1LEVEL"
static PetscErrorCode FETIPJComputeInitialCondition_PJ1LEVEL(FETIPJ ftpj,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  FETI              ft = ftpj->feti;
  PJ1LEVEL          *pj = (PJ1LEVEL*)ftpj->data;
  Vec               asm_e;
  PetscScalar       *rbuff;
  const PetscScalar *sbuff;
  MPI_Comm          comm;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ft,&comm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pj->total_rbm,&asm_e);CHKERRQ(ierr);
  if (ft->n_cs) { ierr = VecGetArrayRead(x,&sbuff);CHKERRQ(ierr);}
  ierr = VecGetArray(asm_e,&rbuff);CHKERRQ(ierr);
  ierr = MPI_Allgatherv(sbuff,ft->n_cs,MPIU_SCALAR,rbuff,pj->count_rbm,pj->displ,MPIU_SCALAR,comm);CHKERRQ(ierr);
  ierr = VecRestoreArray(asm_e,&rbuff);CHKERRQ(ierr);
  if (ft->n_cs) { ierr = VecRestoreArrayRead(x,&sbuff);CHKERRQ(ierr);}
  ierr = FETIPJApplyCoarseProblem_PJ1LEVEL(ftpj,asm_e,y);CHKERRQ(ierr);
  ierr = VecDestroy(&asm_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "FETIPJSetFromOptions_PJ1LEVEL"
/*@
   FETIPJSetFromOptions_FETI1 - Function to set up options from command line.

   Input Parameter:
.  PetscOptionsObject - the PetscOptionItems context
.  ftpj               - the FETIPJ context

   Level: beginner

.keywords: FETIPJ, options
@*/
static PetscErrorCode FETIPJSetFromOptions_PJ1LEVEL(PetscOptionItems *PetscOptionsObject,FETIPJ ftpj)
{
  PetscErrorCode ierr;
  PJ1LEVEL        *pj = (PJ1LEVEL*)ftpj->data;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"FETIPJ 1 level options");CHKERRQ(ierr);

  /* Primal space cumstomization */
  ierr = PetscOptionsBool("-feti_pj1level_destroy_coarse","If set, the matrix of the coarse problem, that is (G^T*G) or (G^T*Q*G), will be destroyed","none",pj->destroy_coarse,&pj->destroy_coarse,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FETIPJCreate_PJ1LEVEL"
PETSC_EXTERN PetscErrorCode FETIPJCreate_PJ1LEVEL(FETIPJ);
PetscErrorCode FETIPJCreate_PJ1LEVEL(FETIPJ ftpj)
{
  PetscErrorCode  ierr;
  FETI            ft = ftpj->feti;
  PJ1LEVEL        *pj = (PJ1LEVEL*)ftpj->data;
  
  PetscFunctionBegin;
  ierr       = PetscNewLog(ft,&pj);CHKERRQ(ierr);
  ftpj->data = (void*)pj;

  ierr  = PetscMemzero(pj,sizeof(PJ1LEVEL));CHKERRQ(ierr);

  ftpj->ops->setup               = FETIPJSetUp_PJ1LEVEL;
  ftpj->ops->destroy             = FETIPJDestroy_PJ1LEVEL;
  ftpj->ops->setfromoptions      = FETIPJSetFromOptions_PJ1LEVEL;
  ftpj->ops->gatherneighbors     = 0; /* FETIPJGatherNeighborsCoarseBasis_PJ1LEVEL; */
  ftpj->ops->assemble            = 0; /* FETIPJAssembleCoarseProblem_PJ1LEVEL; */
  ftpj->ops->factorize           = FETIPJFactorizeCoarseProblem_PJ1LEVEL;
  ftpj->ops->initialcondition    = FETIPJComputeInitialCondition_PJ1LEVEL;
  ftpj->ops->computealpha        = FETIPJComputeAlphaNullSpace_PJ1LEVEL;

  PetscFunctionReturn(0);
}

