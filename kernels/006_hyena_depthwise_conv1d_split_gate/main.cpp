// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
void run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
         const at::Tensor&, const at::Tensor&, const at::Tensor&);
void configured_run(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("configured_run", &configured_run);
}
