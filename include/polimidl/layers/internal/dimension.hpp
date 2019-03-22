#ifndef POLIMIDL_LAYERS_INTERNAL_DIMENSION_HPP
#define POLIMIDL_LAYERS_INTERNAL_DIMENSION_HPP

#include <type_traits>

namespace polimidl {
namespace layers {
namespace internal {
template <std::size_t primary_dimension, std::size_t secondary_dimension,
          typename enabled = void>
struct dimension {};

template <std::size_t primary_dimension, std::size_t secondary_dimension>
struct dimension<primary_dimension, secondary_dimension,
      std::enable_if_t<primary_dimension == secondary_dimension, void>> {
  static constexpr bool are_equal = true;
  static constexpr std::size_t value = primary_dimension;
};

template <std::size_t primary_dimension, std::size_t secondary_dimension>
struct dimension<primary_dimension, secondary_dimension,
      std::enable_if_t<primary_dimension != secondary_dimension, void>> {
  static constexpr bool are_equal = false;
};
}
}
}

#endif // POLIMIDL_LAYERS_INTERNAL_DIMENSION_HPP
