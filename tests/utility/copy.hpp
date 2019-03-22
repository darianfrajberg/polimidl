#ifndef POLIMIDL_TESTS_UTILITY_COPY_HPP
#define POLIMIDL_TESTS_UTILITY_COPY_HPP

#include <polimidl/layers/layer.hpp>

#include <algorithm>

namespace polimidl {
namespace testing {
template <typename type_t, typename components>
class copy : public polimidl::layers::layer<float, components> {
  static_assert(components::are_equal,
                "A copy_layer must have the same number of inputs and outputs");
 public:
  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    std::copy(input.begin(), input.end(), output.begin());
  }
};
}
}

#endif // POLIMIDL_TESTS_UTILITY_COPY_HPP
