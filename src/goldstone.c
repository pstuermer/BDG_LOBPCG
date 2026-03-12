#include "bdg_internal.h"

void bdg_set_goldstone_xi(bdg_t *bdg, f64 xi) {
  bdg->ctx->gold_xi = xi;
}

int bdg_deflate_u1(bdg_t *bdg, f64 tol) {
  BDG_REQUIRE(bdg, BDG_HAS_MU, "bdg_deflate_u1");
  if (bdg->complex_psi0)
    return z_deflate_u1(bdg, tol);
  else
    return d_deflate_u1(bdg, tol);
}

int bdg_deflate_auto(bdg_t *bdg, uint64_t n_check, f64 tol) {
  BDG_REQUIRE(bdg, BDG_HAS_MU, "bdg_deflate_auto");
  if (bdg->complex_psi0)
    return z_deflate_auto(bdg, n_check, tol);
  else
    return d_deflate_auto(bdg, n_check, tol);
}
