static char help[] = "Test the creation of globally unassembled vector\n\n";

#include <eins.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  MPI_Comm               comm;
  PetscInt               rank,idx[5]={0,1,2,3,4},global_indices[5];
  PetscErrorCode         ierr;
  Vec                    v,mpivec,multiplicity,v2;
  PetscScalar            dval,dval2[2],vals[5];
  ISLocalToGlobalMapping mapping;
    
  ierr = EinsInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = MPI_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,5,&multiplicity);CHKERRQ(ierr);
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
    ierr = VecSetValues(v,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    global_indices[0]=1;global_indices[1]=2;global_indices[2]=3;global_indices[3]=4;global_indices[4]=5;
    break;
  case 1:
    ierr = VecSetValue(multiplicity,2,3,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,3,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,4,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,1,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,0,2,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=5;vals[1]=4;vals[2]=3;vals[3]=8;vals[4]=10;
    ierr = VecSetValues(v,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    global_indices[0]=5;global_indices[1]=4;global_indices[2]=3;global_indices[3]=6;global_indices[4]=7;
    break;
  case 2:
    ierr = VecSetValue(multiplicity,2,3,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,3,2,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(multiplicity,4,2,INSERT_VALUES);CHKERRQ(ierr);
    vals[0]=3;vals[1]=5;vals[2]=3;vals[3]=8;vals[4]=10;
    ierr = VecSetValues(v,5,idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    global_indices[0]=0;global_indices[1]=8;global_indices[2]=3;global_indices[3]=6;global_indices[4]=7;
    break;
  }
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(multiplicity);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(multiplicity);CHKERRQ(ierr);
  ierr = VecUnAsmSetMultiplicity(v,multiplicity);CHKERRQ(ierr);
  ierr = VecDestroy(&multiplicity);CHKERRQ(ierr);
  
  ierr = ISLocalToGlobalMappingCreate(comm,1,5,global_indices,PETSC_OWN_POINTER,&mapping);
  ierr = VecUnAsmCreateMPIVec(v,mapping,COMPAT_RULE_AVG,&mpivec);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&mapping);CHKERRQ(ierr);
  VecView(mpivec,PETSC_VIEWER_STDOUT_WORLD);
  /* check vecdot */
  ierr = VecDot(v,v,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nVecDot .......................................... %g \n",dval);
  ierr = VecDot(mpivec,mpivec,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "VecDot mpiVec ................................... %g \n",dval);
  /* check norm L2 */
  ierr = VecNorm(v,NORM_2,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_2 .......................................... %g \n",dval);
  ierr = VecNorm(mpivec,NORM_2,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_2 mpiVec ................................... %g \n",dval);
  /* check norm L1 */
  ierr = VecNorm(v,NORM_1,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_1 .......................................... %g \n",dval);
  ierr = VecNorm(mpivec,NORM_1,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_1 mpiVec ................................... %g \n",dval);
  /* check norm inf */
  ierr = VecNorm(v,NORM_INFINITY,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_inf .......................................... %g \n",dval);
  ierr = VecNorm(mpivec,NORM_INFINITY,&dval);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_inf mpiVec ................................... %g \n",dval);
  /* check norm l1 and l2 */
  ierr = VecNorm(v,NORM_1_AND_2,dval2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nNORM_1_2 .......................................... %g, %g \n",dval2[0],dval2[1]);
  ierr = VecNorm(mpivec,NORM_1_AND_2,dval2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "NORM_1_2 mpiVec ................................... %g, %g \n",dval2[0],dval2[1]);
  /* VecDuplicate, VecCopy and VecView */
  ierr = VecDuplicate(v,&v2);CHKERRQ(ierr);
  ierr = VecCopy(v,v2);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v ................................... \n");
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v2 ................................... \n");
  ierr = VecView(v2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  /* VecScale */
  ierr = VecScale(v2,3);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v2 scaled by 3 ..................... \n");
  ierr = VecView(v2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  /* VecAXPY */
  ierr = VecAXPY(v2,-3,v);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,  "\nPrinting v2 AXPY, must be zero .............. \n");
  ierr = VecView(v2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = EinsFinalize();
  return 0;
}
