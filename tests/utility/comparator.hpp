#ifndef POLIMIDL_TESTS_UTILITY_COMPARATOR_HPP
#define POLIMIDL_TESTS_UTILITY_COMPARATOR_HPP

#include <cmath>

namespace polimidl {
namespace testing {
bool is_close(float a, float b) {
  return round(a * 1000.0f) == round(b * 1000.0f);
}
}
}

#endif // POLIMIDL_TESTS_UTILITY_COMPARATOR_HPP
