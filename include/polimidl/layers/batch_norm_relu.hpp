#ifndef POLIMIDL_LAYERS_BATCH_NORM_RELU_HPP
#define POLIMIDL_LAYERS_BATCH_NORM_RELU_HPP

#include "./internal/batch_norm_relu.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components>
auto batch_norm_relu(const type_t* beta, const type_t* mean,
                     const type_t* fused_gamma_variance_epsilon,
                     type_t min = type_t(0),
                     type_t max = std::numeric_limits<type_t>::max()) {
  static_assert(
      is_components<components>,
      "Invalid 2nd template argument, it should be a polimidl::layers::components");
  static_assert(
      components::are_equal,
      "An batch_norm_relu must have the same number of inputs and outputs");
  return internal::batch_norm_relu<type_t, components>(
      beta, mean, fused_gamma_variance_epsilon, min, max);
}
}
}

#endif // POLIMIDL_LAYERS_BATCH_NORM_RELU_HPP
