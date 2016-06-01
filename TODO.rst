IMPORTANT:
-------------
* Creation and destruction of m_pointer1 can be avoided in the B_cast
  (which is a gather operation for assembling the coarse problem).

* In MPI communications use tags obtained with PetscComm GetNewTag.

* Check the implementation of projection 1 level.

* VecExchange: ISCreate do it as working IS.

* Polling in GENEO implementation can be done without Allreduce
  operations.

* Is it necessary to update RBMs in the case of nonlinear problems.
  
  
NOT SO IMPORTANT:
--------------------
* PJCG: Review case in which you don't specify any projection

* HAVE A LOOK TO THE BUILDING OF MULTIPLICITY OF LAMBDAS

* Take into consideration cases in which none of the subdomains has
  rigid body modes.

* Can FETI1ApplyCoarseProblem_Private() be improved by considering
  VecScatters when mapping from Total_rbm to n_rbm. Right now I'm
  using VecGetSubVector(). Take also a look to
  PCBDDCScatterCoarseDataBegin() in bddcprivate.c

* Consider a reset in class FETI. Should it also be considered in
  Subdomain class?

* SCALING: Is it possible to improve FETIScalingSetUp_multiplicity(),
  specifically line   for ( i=0;i<sd->n_B;i++ ) {
  array[i]=ft->scaling_factor/(sd->count[i]+1);}?
    
* Optimize memory allocation in Subdomain class using PetscMalloc5()?

* Subdomain::vec1_D is not being used by the moment... Try to optimize
  the "work vectors"
  
