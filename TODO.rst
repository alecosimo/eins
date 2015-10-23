* Consider a reset in class FETI. Should it also be considered in
  Subdomain class?

* Check FETIDestroy()

* CHANGE DOC: at the end vec1_global is built during the setup phase
  of the subdomain object.

* SCALING: Is it possible to improve FETIScalingSetUp_multiplicity(),
  specifically line   for ( i=0;i<sd->n_B;i++ ) {
  array[i]=ft->scaling_factor/(sd->count[i]+1);}?
    
* Optimize memory allocation in Subdomain class using PetscMalloc5()?

* Re-think about options in FETI1SetUpNeumannSolver_Private:
  ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PCFactorSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);

* Subdomain::vec1_D is not being used by the moment... Try to optimize
  the "work vectors"

* Functions which are private declare them as static and add _Private
  
