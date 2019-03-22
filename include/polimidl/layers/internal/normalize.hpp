#ifndef POLIMIDL_LAYERS_INTERNAL_NORMALIZE_HPP
#define POLIMIDL_LAYERS_INTERNAL_NORMALIZE_HPP

#include <algorithm>

#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components>
class normalize : public layer<type_t, components, true> {
 public:
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    const auto minmax = std::minmax_element(input.begin(), input.end());
    const auto scale =
        std::max(std::abs(*minmax.first), std::abs(*minmax.second));
    if (scale == type_t(0)) return;
    std::transform(input.begin(), input.end(), output.begin(),
                   [=](type_t value) {
        return value / scale;
    });
  }
};
}
}
}

#endif  // POLIMIDL_LAYERS_NORMALIZE_HPP
