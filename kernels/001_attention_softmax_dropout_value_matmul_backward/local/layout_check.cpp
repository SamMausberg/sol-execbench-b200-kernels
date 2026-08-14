// Host-only CuTe layout verification for the K1 dual-view smem trick.
// Checks that the K-major SW128 layout of a [128 x 128] bf16 tile and the
// MN-major SW128 layout of its transposed view map (i,j) <-> (j,i) to the
// same physical smem offsets, so one TMA-filled buffer can serve as
// MMA2-A (K-major) and MMA1-B (MN-major).
#include <cstdio>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>

using namespace cute;

int main() {
  using T = cutlass::bfloat16_t;
  // Physical buffer layout: filled by TMA as (Q=128, D=128) K-major (D contiguous)
  auto kmaj = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<T>{},
      make_shape(make_shape(Int<128>{}, Int<16>{}), Int<1>{}, Int<8>{}));
  // Desired MMA1-B view: (D=128, Q=128) MN-major (D "MN"-contiguous)
  auto mnmaj = UMMA::tile_to_mma_shape(
      UMMA::Layout_MN_SW128_Atom<T>{},
      make_shape(make_shape(Int<128>{}, Int<16>{}), Int<1>{}, Int<8>{}));
  print("K-major :  "); print(kmaj);  print("\n");
  print("MN-major:  "); print(mnmaj); print("\n");
  long mismatches = 0;
  for (int q = 0; q < 128; q++)
    for (int d = 0; d < 128; d++) {
      auto a = kmaj(make_coord(q, d));
      auto b = mnmaj(make_coord(d, q));
      if (a != b) {
        if (mismatches < 5) printf("MISMATCH q=%d d=%d  kmaj=%d mnmaj=%d\n", q, d, int(a), int(b));
        mismatches++;
      }
    }
  printf("mismatches: %ld\n", mismatches);
  return mismatches != 0;
}
