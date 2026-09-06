// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
void run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
         const at::Tensor&, const at::Tensor&, const at::Tensor&);
void configured_run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&, int, int);
void run_scalar(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&);
void configured_scalar_run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, const at::Tensor&, const at::Tensor&, int, int);
void run_direct(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&);
void configured_direct_run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, const at::Tensor&, const at::Tensor&, int, int);
void run_plain(const at::Tensor&, const at::Tensor&, const at::Tensor&,
               const at::Tensor&, const at::Tensor&, const at::Tensor&);
void configured_plain_run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, const at::Tensor&, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("run_scalar", &run_scalar);
  m.def("configured_scalar_run", &configured_scalar_run);
  m.def("configured_run", &configured_run);
  m.def("run_direct", &run_direct);
  m.def("configured_direct_run", &configured_direct_run);
  m.def("run_plain", &run_plain);
  m.def("configured_plain_run", &configured_plain_run);
}
