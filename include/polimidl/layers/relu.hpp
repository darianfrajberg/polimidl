#ifndef POLIMIDL_LAYERS_RELU_HPP
#define POLIMIDL_LAYERS_RELU_HPP

#include "./internal/relu.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components>
auto relu(type_t min = type_t(0),
          type_t max = std::numeric_limits<type_t>::max()) {
  static_assert(
      is_components<components>,
      "Invalid 2nd template argument, it should be a polimidl::layers::components");
  static_assert(
      components::are_equal,
      "An relu must have the same number of inputs and outputs");
  return internal::relu<type_t, components>(min, max);
}
}
}

#endif  // POLIMIDL_LAYERS_RELU_HPP
