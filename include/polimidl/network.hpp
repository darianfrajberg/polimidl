#ifndef POLIMIDL_NETWORK_HPP
#define POLIMIDL_NETWORK_HPP

#include <memory>

#include "internal/network.hpp"
#include "internal/runner.hpp"

namespace polimidl {
unsigned int max_number_of_workers() {
  return std::thread::hardware_concurrency();
}

unsigned int suggested_number_of_workers() {
  return std::max(static_cast<unsigned int>(1),
                  std::thread::hardware_concurrency() - 1);
}

template <typename type_t, typename... Layers>
internal::network<type_t, Layers...> build_network(
    std::size_t rows, std::size_t columns, unsigned int number_of_workers,
    Layers&&... layers) {
  return internal::network<type_t, Layers...>(rows, columns, number_of_workers,
                                              std::forward<Layers>(layers)...);
}

template <typename type_t, typename... Layers>
std::unique_ptr<internal::network<type_t, Layers...>> make_network(
    std::size_t rows, std::size_t columns, unsigned int number_of_workers,
    Layers&&... layers) {
  return std::make_unique<internal::network<type_t, Layers...>>(
    rows, columns, number_of_workers, std::forward<Layers>(layers)...);
}
}

#endif // POLIMIDL_NETWORK_HPP
