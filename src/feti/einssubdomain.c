if (!feti->subdomain) SETERRQ(PetscObjectComm((PetscObject)feti),PETSC_ERR_ARG_WRONGSTATE,"Subdomain must be set first");
