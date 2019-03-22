#include <polimidl/network.hpp>
#include <polimidl/layers/relu.hpp>
#include <iostream>
#include "../utility/comparator.hpp"

int test(unsigned int number_of_workers){
  using polimidl::build_network;
  using polimidl::layers::relu;
  using polimidl::layers::components;
  using polimidl::testing::is_close;
  constexpr std::size_t h = 2;
  constexpr std::size_t w = 2;
  constexpr std::size_t channels = 3;
  constexpr float input[] = {0.5,-4,8,-1,5.5,9,-2,6.5,-10,3.5,7,11}; //2x2x3

  {
    constexpr float expected_output[] = {0.5,0,8,0,5.5,9,0,6.5,0,3.5,7,11};

    auto net = build_network<float>(h, w, number_of_workers,
        relu<float, components<channels>>()
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
      std::cerr << "Invalid input shape for relu: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
      return -1;
    }
    if (net.input().size() != std::size(input)) {
      std::cerr << "Invalid input size for relu: " <<
          net.input().size() << std::endl;
      return -1;
    }
    if (net.output_rows() != h || net.output_columns() != w || net.output_components() != channels) {
      std::cerr << "Invalid output shape for relu: " <<
      net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
      std::cerr << "Invalid output size for relu: " <<
          output.size() << std::endl;
      return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr << "Invalid output value for relu" <<
          std::endl;
      return -1;
    }
  }

  {
    constexpr float expected_output[] = {0.5,0,6,0,5.5,6,0,6,0,3.5,6,6};

    auto net = build_network<float>(h, w, number_of_workers,
        relu<float, components<channels>>(/* min */ 0, /* max */ 6)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
      std::cerr << "Invalid input shape for relu6: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
      return -1;
    }
    if (net.input().size() != std::size(input)) {
      std::cerr << "Invalid input size for relu6: " <<
          net.input().size() << std::endl;
      return -1;
    }
    if (net.output_rows() != h || net.output_columns() != w || net.output_components() != channels) {
      std::cerr << "Invalid output shape for relu6: " <<
      net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
      std::cerr << "Invalid output size for relu6: " <<
          output.size() << std::endl;
      return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr << "Invalid output value for relu6" <<
          std::endl;
      return -1;
    }
  }
  return 0;
}

int main() {
    using polimidl::suggested_number_of_workers;
    int value = test(1);
    if (value) return value;
    value = test(suggested_number_of_workers());
    if (value) return value;
    std::cout << "Passed test" << std::endl;
    return 0;
}
