#include <einsvec.h>
#include <einsmat.h>
#include <petsc/private/petscimpl.h>


struct _LGMat_ctx {
  Mat  localA;
};
typedef struct _LGMat_ctx *LGMat_ctx;
