static char help[] = "Test the creation of globally unassembled vector\n\n";

#include <eins.h>
#include <einstest.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  MPI_Comm               comm;
  PetscInt               rank,idx[5]={0,1,2,3,4},*global_indices;
  PetscErrorCode         ierr;
  Vec                    v,mpivec,multiplicity,v2,refvec,vec_comp,v3;
  PetscScalar            dval0,dval1,dval2_0[2],dval2_1[2],vals[5];
  ISLocalToGlobalMapping mapping;
  VecExchange            ve;
  PetscInt               n_neigh;
  PetscInt               *neigh, *n_shared, **shared;
  
  ierr = EinsInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = MPI_COMM_WORLD;
  ierr = PetscMalloc1(5,&global_indices);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,5,&multiplicity);CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,5,&vec_comp);CHKERRQ(ierr);
  ierr = VecCreate(comm,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,5,9);CHKERRQ(ierr);
  ierr = VecSetType(v,VECMPIUNASM);CHKERRQ(ierr);
  ierr = VecSet(multiplicity,1);CHKERRQ(ierr);
  switch (rank){
  case 0:
    ierr = VecSetValue(multiplicity,2,3,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,3,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,4,2,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=1;vals[1]=2;vals[2]=3;vals[3]=4;vals[4]=5;
    ierr = VecSetValuesLocal(v,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=1;vals[1]=2;vals[2]=9;vals[3]=8;vals[4]=10;
    ierr = VecSetValues(vec_comp,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    global_indices[0]=1;global_indices[1]=2;global_indices[2]=3;global_indices[3]=4;global_indices[4]=5;
    break;
  case 1:
    ierr = VecSetValue(multiplicity,2,3,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,3,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,4,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,1,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,0,2,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=5;vals[1]=4;vals[2]=3;vals[3]=8;vals[4]=10;
    ierr = VecSetValuesLocal(v,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=10;vals[1]=8;vals[2]=9;vals[3]=16;vals[4]=20;
    ierr = VecSetValues(vec_comp,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    global_indices[0]=5;global_indices[1]=4;global_indices[2]=3;global_indices[3]=6;global_indices[4]=7;
    break;
  case 2:
    ierr = VecSetValue(multiplicity,2,3,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,3,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,4,2,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=3;vals[1]=5;vals[2]=3;vals[3]=8;vals[4]=10;
    ierr = VecSetValuesLocal(v,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=3;vals[1]=5;vals[2]=9;vals[3]=16;vals[4]=20;
    ierr = VecSetValues(vec_comp,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    global_indices[0]=0;global_indices[1]=8;global_indices[2]=3;global_indices[3]=6;global_indices[4]=7;
    break;
  }
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vec_comp);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec_comp);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(multiplicity);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(multiplicity);CHKERRQ(ierr);
  ierr = VecUnAsmSetMultiplicity(v,multiplicity);CHKERRQ(ierr);
  ierr = VecDestroy(&multiplicity);CHKERRQ(ierr);
  
  ierr = ISLocalToGlobalMappingCreate(comm,1,5,global_indices,PETSC_OWN_POINTER,&mapping);
  ierr = VecUnAsmCreateMPIVec(v,mapping,COMPAT_RULE_AVG,&mpivec);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr); 
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = VecView(mpivec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* check vecdot */
  ierr = VecDot(v,v,&dval0);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nVecDot .......................................... %e \n",dval0);
  ierr = VecDot(mpivec,mpivec,&dval1);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "VecDot mpiVec ................................... %e \n",dval1);
  ierr = TestAssertScalars(dval0,dval1,1e-8);CHKERRQ(ierr);   
  /* check norm L2 */
  ierr = VecNorm(v,NORM_2,&dval0);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_2 .......................................... %e \n",dval0);
  ierr = VecNorm(mpivec,NORM_2,&dval1);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_2 mpiVec ................................... %e \n",dval1);
  ierr = TestAssertScalars(dval0,dval1,1e-8);CHKERRQ(ierr);   
  /* check norm L1 */
  ierr = VecNorm(v,NORM_1,&dval1);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_1 .......................................... %e \n",dval1);
  ierr = VecNorm(mpivec,NORM_1,&dval0);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_1 mpiVec ................................... %e \n",dval0);
  ierr = TestAssertScalars(dval0,dval1,1e-8);CHKERRQ(ierr);   
  /* check norm inf */
  ierr = VecNorm(v,NORM_INFINITY,&dval0);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_inf .......................................... %e \n",dval0);
  ierr = VecNorm(mpivec,NORM_INFINITY,&dval1);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_inf mpiVec ................................... %e \n",dval1);
  ierr = TestAssertScalars(dval0,dval1,1e-8);CHKERRQ(ierr);   
  /* check norm l1 and l2 */
  ierr = VecNorm(v,NORM_1_AND_2,dval2_0);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_1_2 .......................................... %e, %e \n",dval2_0[0],dval2_0[1]);
  ierr = VecNorm(mpivec,NORM_1_AND_2,dval2_1);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_1_2 mpiVec ................................... %e, %e \n",dval2_1[0],dval2_1[1]);
  ierr = TestAssertScalars(dval2_0[0],dval2_1[0],1e-8);CHKERRQ(ierr);
  ierr = TestAssertScalars(dval2_0[1],dval2_1[1],1e-8);CHKERRQ(ierr);
  /* VecDuplicate, VecCopy and VecView */
  ierr = VecDuplicate(v,&v2);CHKERRQ(ierr);
  ierr = VecCopy(v,v2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v ................................... \n");
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v2 ................................... \n");
  ierr = VecView(v2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TestAssertVectors(v,v2,1e-8);CHKERRQ(ierr);
  /* VecScale */
  ierr = VecScale(v2,3);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v2 scaled by 3 ..................... \n");
  ierr = VecView(v2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* VecWAXPY */
  ierr = VecDuplicate(v,&v3);CHKERRQ(ierr);
  ierr = VecCopy(v,v3);CHKERRQ(ierr);
  ierr = VecAXPBY(v3,-1,2,v2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v3 WAXPY, must be zero .............. \n");
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(v2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(v3,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = TestAssertVectorLeTol(v3,1e-8);CHKERRQ(ierr);
  /* VecAXPY */
  ierr = VecAXPY(v2,-3,v);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v2 AXPY, must be zero .............. \n");
  ierr = VecView(v2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TestAssertVectorLeTol(v2,1e-8);CHKERRQ(ierr);
  /* VecExchange */
  ierr = ISLocalToGlobalMappingGetInfo(mapping,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr); 
  ierr = VecExchangeCreate(v,n_neigh,neigh,n_shared,shared,PETSC_USE_POINTER,&ve);CHKERRQ(ierr);
  ierr = VecExchangeBegin(ve,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecExchangeEnd(ve,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecUnAsmGetLocalVector(v,&refvec);CHKERRQ(ierr);
  ierr = TestAssertVectors(refvec,vec_comp,1e-8);CHKERRQ(ierr);
  
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting VecExchange .............. \n");
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  ierr = ISLocalToGlobalMappingRestoreInfo(mapping,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);
  ierr = VecExchangeDestroy(&ve);CHKERRQ(ierr);
  ierr = VecDestroy(&v2);CHKERRQ(ierr);
  ierr = VecDestroy(&v3);CHKERRQ(ierr);
  ierr = VecDestroy(&vec_comp);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&mpivec);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr); 
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
  ierr = EinsFinalize();CHKERRQ(ierr);
  return 0;
}
