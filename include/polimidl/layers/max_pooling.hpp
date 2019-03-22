#ifndef POLIMIDL_LAYERS_MAX_POOLING_HPP
#define POLIMIDL_LAYERS_MAX_POOLING_HPP

#include "./internal/max_pooling.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components, typename kernel,
          typename stride = stride<1>>
auto max_pooling() {
  static_assert(
      is_components<components>,
      "Invalid 2nd template argument, it should be a polimidl::layers::components");
  static_assert(
      is_kernel<kernel>,
      "Invalid 3rd template argument, it should be a polimidl::layers::kernel");
  static_assert(
      is_stride<stride>,
      "Invalid 4th template argument, it should be a polimidl::layers::stride");
  static_assert(
      components::are_equal,
      "An max_pooling must have the same number of inputs and outputs");
  static_assert(
      !kernel::is_pointwise,
      "An max_pooling must have a non pointwise kernel (1x1)");
  return internal::max_pooling<type_t, components, kernel, stride>();
}
}
}

#endif // POLIMIDL_LAYERS_MAX_POOLING_HPP
