#include <iostream>
#include <polimidl/network.hpp>

#include "../utility/noop.hpp"

int main() {
  using polimidl::build_network;
  using polimidl::make_network;
  using polimidl::layers::components;
  using polimidl::suggested_number_of_workers;
  using polimidl::testing::noop;
  {
    auto network = build_network<float>(1, 1, suggested_number_of_workers(),
                                        noop<float, components<1>>());
  }
  {
    auto network = make_network<float>(1, 1, suggested_number_of_workers(),
                                       noop<float, components<1>>());
  }
  std::cout << "Passed Test";
}
