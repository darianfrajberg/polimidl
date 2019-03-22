#ifndef POLIMIDL_LAYERS_DIMENSIONS_HPP
#define POLIMIDL_LAYERS_DIMENSIONS_HPP

#include <type_traits>

#include "./internal/dimension.hpp"

namespace polimidl {
namespace layers {
template <std::size_t input_components, std::size_t output_components = input_components>
struct components :
    public internal::dimension<input_components, output_components> {
  static_assert(input_components > 0 && output_components > 0,
                "Components must be grater than 0");
  static constexpr std::size_t input = input_components;
  static constexpr std::size_t output = output_components;
  static constexpr bool are_increasing = output > input;
  static constexpr bool are_reducing = input > output;
};

template <typename type_t>
inline constexpr bool is_components =
    std::is_same_v<type_t, components<type_t::input, type_t::output>>;

template <std::size_t rows_stride, std::size_t columns_stride = rows_stride>
struct stride :
    public internal::dimension<rows_stride, columns_stride> {
  static_assert(rows_stride > 0 && columns_stride > 0,
                "Stride must be grater than 0");
  using dimension_t = internal::dimension<rows_stride, columns_stride>;
  static constexpr std::size_t rows = rows_stride;
  static constexpr std::size_t columns = columns_stride;
  static constexpr bool no_stride =
      dimension_t::are_equal && dimension_t::value == 1;
};

template <typename type_t>
inline constexpr bool is_stride =
    std::is_same_v<type_t, stride<type_t::rows, type_t::columns>>;

template <std::size_t kernel_rows, std::size_t kernel_columns = kernel_rows>
struct kernel :
    public internal::dimension<kernel_rows, kernel_columns> {
  static_assert(kernel_rows > 0 && kernel_columns > 0,
                "Kernel size must be grater than 0");
  using dimension_t = internal::dimension<kernel_rows, kernel_columns>;
  static constexpr std::size_t rows = kernel_rows;
  static constexpr std::size_t columns = kernel_columns;
  static constexpr bool is_pointwise =
      dimension_t::are_equal && dimension_t::value == 1;
};

template <typename type_t>
inline constexpr bool is_kernel =
    std::is_same_v<type_t, kernel<type_t::rows, type_t::columns>>;

template <std::size_t pooling_rows, std::size_t pooling_columns = pooling_rows>
struct pooling :
    public internal::dimension<pooling_rows, pooling_columns> {
  static_assert(pooling_rows > 1 && pooling_columns > 1,
                "Kernel size must be grater than 1");
  static constexpr std::size_t rows = pooling_rows;
  static constexpr std::size_t columns = pooling_columns;
};

template <typename type_t>
inline constexpr bool is_pooling =
    std::is_same_v<type_t, pooling<type_t::rows, type_t::columns>>;

template <std::size_t padding_rows, std::size_t padding_columns = padding_rows>
struct padding :
    public internal::dimension<padding_rows, padding_columns> {
  using dimension_t = internal::dimension<padding_rows, padding_columns>;
  static constexpr std::size_t rows = padding_rows;
  static constexpr std::size_t columns = padding_columns;
  static constexpr std::size_t top = rows / 2;
  static constexpr std::size_t left = columns / 2;
  static constexpr std::size_t bottom = top + rows % 2;
  static constexpr std::size_t right = left + columns % 2;
  static constexpr bool no_padding =
      dimension_t::are_equal && dimension_t::value == 0;
};

template <typename type_t>
inline constexpr bool is_padding =
    std::is_same_v<type_t, padding<type_t::rows, type_t::columns>>;
}
}

#endif // POLIMIDL_LAYERS_DIMENSIONS_HPP
