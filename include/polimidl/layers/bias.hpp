#ifndef POLIMIDL_LAYERS_BIAS_HPP
#define POLIMIDL_LAYERS_BIAS_HPP

#include "./internal/bias.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components>
auto bias(const type_t* bias) {
  static_assert(
      is_components<components>,
      "Invalid 2nd template argument, it should be a polimidl::layers::components");
  static_assert(
      components::are_equal,
      "An bias must have the same number of inputs and outputs");
  return internal::bias<type_t, components>(bias);
}
}
}

#endif  // POLIMIDL_LAYERS_BIAS_HPP
