#ifndef POLIMIDL_LAYERS_BATCH_NORM_HPP
#define POLIMIDL_LAYERS_BATCH_NORM_HPP

#include "./internal/batch_norm.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components>
auto batch_norm(const type_t* beta, const type_t* mean,
                const type_t* fused_gamma_variance_epsilon) {
  static_assert(
      is_components<components>,
      "Invalid 2nd template argument, it should be a polimidl::layers::components");
  static_assert(
      components::are_equal,
      "An batch_norm_relu must have the same number of inputs and outputs");

  return internal::batch_norm<type_t, components>(
      beta, mean, fused_gamma_variance_epsilon);
}
}
}

#endif // POLIMIDL_LAYERS_BATCH_NORM_HPP
