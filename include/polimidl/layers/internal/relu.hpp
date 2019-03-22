#ifndef POLIMIDL_LAYERS_INTERNAL_RELU_HPP
#define POLIMIDL_LAYERS_INTERNAL_RELU_HPP

#include <algorithm>

#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components>
class relu : public layer<type_t, components, true> {
 public:
  relu(type_t min, type_t max) : min_(min), max_(max) {}

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    std::transform(input.begin(), input.end(), output.begin(),
                   [=](type_t value) {
        return std::clamp(value, min_, max_);
    });
  }

 private:
  const type_t min_;
  const type_t max_;
};
}
}
}

#endif  // POLIMIDL_LAYERS_INTERNAL_RELU_HPP
