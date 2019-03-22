#ifndef POLIMIDL_TESTS_UTILITY_NOOP_HPP
#define POLIMIDL_TESTS_UTILITY_NOOP_HPP

#include <polimidl/layers/layer.hpp>

namespace polimidl {
namespace testing {
template <typename type_t, typename components>
class noop : public polimidl::layers::layer<float, components> {
  static_assert(components::are_equal,
                "A noop must have the same number of inputs and outputs");
};
}
}

#endif // POLIMIDL_TESTS_UTILITY_NOOP_HPP
