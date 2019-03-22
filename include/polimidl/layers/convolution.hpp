#ifndef POLIMIDL_LAYERS_CONVOLUTION_HPP
#define POLIMIDL_LAYERS_CONVOLUTION_HPP

#include "./internal/convolution.hpp"
#include "./internal/pointwise_convolution.hpp"
#include "./internal/pointwise_convolution_inplace.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components, typename kernel,
          typename stride = stride<1>, typename padding = padding<0>>
auto convolution(const type_t* coeff) {
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
      is_padding<padding>,
      "Invalid 5th template argument, it should be a polimidl::layers::padding");
  if constexpr (kernel::is_pointwise && stride::no_stride
                && padding::no_padding) {
    if constexpr (components::are_equal || components::are_reducing) {
      return internal::pointwise_convolution_inplace<type_t, components>(coeff);
    } else {
      return internal::pointwise_convolution<type_t, components>(coeff);
    }
  } else {
    return internal::convolution<
      type_t, components, kernel, stride, padding>(coeff);
  }
}
}
}

#endif  // POLIMIDL_LAYERS_CONVOLUTION_HPP
