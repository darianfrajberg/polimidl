#ifndef POLIMIDL_LAYERS_INTERNAL_SOFTMAX_HPP
#define POLIMIDL_LAYERS_INTERNAL_SOFTMAX_HPP

#include <algorithm>

#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components>
class softmax : public layer<type_t, components, true> {
 public:
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    for (auto current = input.begin(); current != input.end();
         current += components::value){
        auto end_current = current + components::value;
        type_t max = *std::max_element(current, end_current);
        type_t sum = type_t(0);
        std::transform(current, end_current, current, [=, &sum](type_t value) {
            value = std::exp(value - max);
            sum += value;
            return value;
        });
        std::transform(current, end_current, current, [=](type_t value) {
            return value / sum;
        });
    }
  }
};
}
}
}

#endif  // POLIMIDL_LAYERS_INTERNAL_SOFTMAX_HPP
