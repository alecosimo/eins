* Consider a reset in class FETI. Should it also be considered in
  Subdomain class?

* Check FETIDestroy()

* SCALING: take into account that part of the scaling of BDDC is
  taking part at the end PCISSetUp

* CHANGE DOC: at the end vec1_global is built during the setup phase
  of the subdomain object.

* SCALING: take into account that part of the scaling is set up in
  bddcfetidp.c::PCBDDCSetupFETIDPMatContext
  
* Optimize memory allocation in Subdomain class using PetscMalloc5()?

* Re-think about options in FETI1SetUpNeumannSolver_Private:
  ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PCFactorSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);

* Subdomain::vec1_D is not being used by the moment... Try to optimize
  the "work vectors"
