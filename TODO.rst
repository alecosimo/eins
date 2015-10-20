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
