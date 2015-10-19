* Consider a reset in class FETI. Should it also be considered in
  Subdomain class?

* Check FETIDestroy()

* Add FETIRegisterAll() to EinsInitialize()

* SCALING: take into account that part of the scaling of BDDC is
  taking part at the end PCISSetUp

* IMPORTANT: global_to_B ---> in feti we do not need to work with
  global primal quantities. So, by the moment this structure can be
  avoided. Maybe, it could be useful for building the solution on a
  global distributed vector.... Regarding the simple petsc object
  Subdomain: is it really necessary? I think it could be interesting
  for in a future supporting other type of "substructuring" methods.

* CHANGE DOC: at the end vec1_global is built during the setup phase
  of the subdomain object.

* SCALING: take into account that part of the scaling is set up in
  bddcfetidp.c::PCBDDCSetupFETIDPMatContext
  
* Optimize memory allocation in Subdomain class using PetscMalloc5()?
