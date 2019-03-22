#include <algorithm>
#include <iostream>
#include <polimidl/network.hpp>
#include <polimidl/layers/avg_pooling.hpp>
#include "../utility/comparator.hpp"

int test(unsigned int number_of_workers){
  using polimidl::build_network;
  using polimidl::layers::avg_pooling;
  using polimidl::layers::components;
  using polimidl::layers::kernel;
  using polimidl::layers::stride;
  using polimidl::testing::is_close;
  constexpr std::size_t h = 3;
  constexpr std::size_t w = 3;
  constexpr std::size_t channels = 2;
  constexpr float input[] = {0,9,1,10,2,11,3,12,4,13,5,14,6,15,7,16,8,17}; //3x3x2

  {
    constexpr float expected_output[] = {2, 11, 3, 12, 5, 14, 6, 15};
    auto net = build_network<float>(h, w, number_of_workers,
        avg_pooling<float, components<channels>, kernel<2>>()
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for avg pooling with stride 1: " <<
            net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for avg pooling with stride 1: " <<
            net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != channels) {
        std::cerr << "Invalid output shape for avg pooling with stride 1: " <<
            net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
        return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for avg pooling with stride 1: " <<
            output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                    std::begin(expected_output), std::end(expected_output),
                    is_close)){
      std::cerr << "Invalid output value for avg pooling with stride 1" <<
          std::endl;
      return -1;
    }
  }

  {
    constexpr float expected_output[] = {2, 11};
    auto net = build_network<float>(h, w, number_of_workers,
      avg_pooling<float, components<channels>, kernel<2>, stride<2>>()
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for avg pooling with stride 2: " <<
            net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for avg pooling with stride 2: " <<
            net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 1 || net.output_columns() != 1 || net.output_components() != channels) {
        std::cerr << "Invalid output shape for avg pooling with stride 2: " <<
            net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
        return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for avg pooling with stride 2: " <<
            output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                    std::begin(expected_output), std::end(expected_output),
                    is_close)){
      std::cerr << "Invalid output value for avg pooling with stride 2" <<
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
