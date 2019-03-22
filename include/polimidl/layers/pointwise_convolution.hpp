#ifndef POLIMIDL_LAYERS_POINTWISE_CONVOLUTION_HPP
#define POLIMIDL_LAYERS_POINTWISE_CONVOLUTION_HPP

#include "./convolution.hpp"

namespace polimidl {
namespace layers {
template <typename type_t, typename components,
          typename stride = stride<1>, typename padding = padding<0>>
auto pointwise_convolution(const type_t* coeff) {
  static_assert(
      is_components<components>,
      "Invalid 2nd template argument, it should be a polimidl::layers::components");
  static_assert(
      is_stride<stride>,
      "Invalid 2th template argument, it should be a polimidl::layers::stride");
  static_assert(
      is_padding<padding>,
      "Invalid 3th template argument, it should be a polimidl::layers::padding");
  return convolution<type_t, components, kernel<1>, stride, padding>(coeff);
}
}
}

#endif  // POLIMIDL_LAYERS_POINTWISE_CONVOLUTION_HPP
