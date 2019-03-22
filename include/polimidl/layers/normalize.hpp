#ifndef POLIMIDL_LAYERS_NORMALIZE_HPP
#define POLIMIDL_LAYERS_NORMALIZE_HPP

#include "./internal/normalize.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components>
auto normalize {
  static_assert(
      is_components<components>,
      "Invalid 2nd template argument, it should be a polimidl::layers::components");
  static_assert(
      components::are_equal,
      "A normalize must have the same number of inputs and outputs");
  return internal::normalize<type_t, components>();
}
}
}

#endif  // POLIMIDL_LAYERS_NORMALIZE_HPP
