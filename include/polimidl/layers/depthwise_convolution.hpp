#ifndef POLIMIDL_LAYERS_DEPTHWISE_CONVOLUTION_HPP
#define POLIMIDL_LAYERS_DEPTHWISE_CONVOLUTION_HPP

#include "./internal/depthwise_convolution.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components, typename kernel,
          typename stride = stride<1>, typename padding = padding<0>>
auto depthwise_convolution(type_t const *coeff) {
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
  static_assert(
      components::are_equal,
      "A depthwise_convolution must have the same number of inputs and outputs");
  return internal::depthwise_convolution<
      type_t, components, kernel, stride, padding>(coeff);
}
}
}

#endif  // POLIMIDL_LAYERS_DEPTHWISE_CONVOLUTION_HPP
