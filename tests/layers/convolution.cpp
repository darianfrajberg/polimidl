#include <polimidl/network.hpp>
#include <polimidl/layers/convolution.hpp>
#include <polimidl/layers/alignment.hpp>
#include <iostream>
#include "../utility/comparator.hpp"
#include "../utility/copy.hpp"

int test(unsigned int number_of_workers){
  using polimidl::build_network;
  using polimidl::layers::components;
  using polimidl::layers::convolution;
  using polimidl::layers::kernel;
  using polimidl::layers::stride;
  using polimidl::testing::copy;
  using polimidl::testing::is_close;
  using polimidl::layers::buffer_alignment;
  const std::size_t h = 4;
  const std::size_t w = 4;
  constexpr std::size_t channels = 1;
  const float input[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}; //4x4x1
  alignas(buffer_alignment::byte_alignment) const float weights[] = {0,1,2,3}; //2x2x1

  {
    const float expected_output[] = {24, 30, 36, 48, 54, 60, 72, 78, 84};
    auto net = build_network<float>(h, w, number_of_workers,
      convolution<float, components<channels, 1>, kernel<2>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for convolution with stride 1: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for convolution with stride 1: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 3 || net.output_columns() != 3 || net.output_components() != channels) {
        std::cerr << "Invalid output shape for convolution with stride 1: " <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for convolution with stride 1: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr << "Invalid output value for convolution with stride 1" <<
          std::endl;
      return -1;
    }
  }

  {
    const float expected_output[] = {24, 30, 36, 48, 54, 60, 72, 78, 84};
    auto net = build_network<float>(h, w, number_of_workers,
      copy<float, components<channels>>(),
      convolution<float, components<channels, 1>, kernel<2>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for inverted convolution with stride 1: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for inverted convolution with stride 1: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 3 || net.output_columns() != 3 || net.output_components() != channels) {
      std::cerr << "Invalid output shape for inverted convolution with stride 1: " <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for inverted convolution with stride 1: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr << "Invalid output value for inverted convolution with stride 1" <<
          std::endl;
      return -1;
    }
  }

  {
    const float expected_output[] = {24,36,72,84};
    auto net = build_network<float>(h, w, number_of_workers,
      convolution<float, components<channels, 1>, kernel<2>, stride<2>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for convolution with stride 2: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for convolution with stride 2: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != channels) {
      std::cerr << "Invalid output shape for convolution with stride 2: " <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for convolution with stride 2: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr << "Invalid output value for convolution with stride 2" <<
          std::endl;
      return -1;
    }
  }

  {
    const float expected_output[] = {24,36,72,84};
    auto net = build_network<float>(h, w, number_of_workers,
      copy<float, components<channels>>(),
      convolution<float, components<channels, 1>, kernel<2>, stride<2>>(weights)
    );
    if (net.input_rows() != h || net.input_columns() != w || net.input_components() != channels) {
        std::cerr << "Invalid input shape for inverted convolution with stride 2: " <<
        net.input_rows() << "x" << net.input_columns() << "x" << net.input_components() << std::endl;
        return -1;
    }
    if (net.input().size() != std::size(input)) {
        std::cerr << "Invalid input size for inverted convolution with stride 2: " <<
        net.input().size() << std::endl;
        return -1;
    }
    if (net.output_rows() != 2 || net.output_columns() != 2 || net.output_components() != channels) {
      std::cerr << "Invalid output shape for inverted convolution with stride 2: " <<
          net.output_rows() << "x" << net.output_columns() << "x" << net.output_components() << std::endl;
      return -1;
    }
    std::copy(std::begin(input), std::end(input), net.input().begin());
    auto output = net();
    if (output.size() != std::size(expected_output)) {
        std::cerr << "Invalid output size for inverted convolution with stride 2: " <<
        output.size() << std::endl;
        return -1;
    }
    if (!std::equal(output.begin(), output.end(),
                   std::begin(expected_output), std::end(expected_output),
                   is_close)){
      std::cerr << "Invalid output value for inverted convolution with stride 2" <<
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

