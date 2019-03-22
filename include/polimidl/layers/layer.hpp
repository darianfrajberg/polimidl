#ifndef POLIMIDL_LAYERS_LAYER_HPP
#define POLIMIDL_LAYERS_LAYER_HPP

#include "./dimensions.hpp"

namespace polimidl {
namespace layers {
template <typename data_type_t, typename components, bool in_place = false>
class layer {
 public:
  static_assert(
    is_components<components>,
    "The components of a layer should be a polimidl::layers::components");
  using type_t = data_type_t;
  static constexpr std::size_t input_components = components::input;
  static constexpr std::size_t output_components = components::output;
  static constexpr bool is_in_place = in_place;

  static constexpr std::size_t output_rows(std::size_t input_rows) {
    return input_rows;
  }
  static constexpr std::size_t output_columns(std::size_t input_columns) {
    return input_columns;
  }
  static constexpr std::size_t temporary_size(std::size_t input_rows,
                                         std::size_t input_columns,
                                         unsigned int number_of_workers) {
    return 0;
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void optimize_for(input_t input, temporary_t temp, output_t output,
                    std::size_t input_rows, std::size_t input_columns,
                    const scheduler_t& scheduler) const {}
};
}
}

#endif // POLIMIDL_LAYERS_LAYER_HPP
