#include <polimidl/network.hpp>
#include <iostream>
#include "../../examples/mobilenet_inputs/input_egyptian_cat.h" // normalized input 224x224x3 of egyptian cat
#include "../../examples/mobilenet_inputs/input_dog.h" // normalized input 224x224x3 of dog
#include "../../examples/mobilenet.hpp"
#include "../utility/comparator.hpp"

int test(unsigned int number_of_workers){
  using polimidl::testing::is_close;
  constexpr int h = 224, w = 224, channels = 3, categories = 1001;
  {
    constexpr float expected_index = 286; // Egyptian cat
    constexpr float expected_value = 0.80088;
    auto net = build_mobilenet(h, w, number_of_workers);
    if (net.input_rows() != 224 || net.input_columns() != 224 || net.input_components() != channels) {
      std::cerr << "Invalid input shape for mobilenet 1: "<< net.input_rows() <<
          "x" << net.input_columns() << "x" << net.input_components() <<std::endl;
      return -1;
    }
    if (net.input().size() != 150528) {
      std::cerr << "Invalid input size for mobilenet 1: " << net.input().size()
          << std::endl;
      return -1;
    }
    if (net.output_rows() != 1 || net.output_columns() != 1 || net.output_components() != categories) {
        std::cerr << "Invalid output shape for mobilenet 1: "<< net.output_rows() <<
        "x" << net.output_columns() <<std::endl;
        return -1;
    }
    std::copy(std::begin(input_egyptian_cat), std::end(input_egyptian_cat),
              net.input().begin());
    auto output = net();
    if (output.size() != categories) {
      std::cerr << "Invalid output size for mobilenet 1: " << output.size()
          << std::endl;
      return -1;
    }
    auto max_addr = std::max_element(output.begin(), output.end());
    float max_value = *max_addr;
    int max_index = max_addr - output.begin();
    if (max_index != expected_index){
      std::cerr << "Invalid output for expected Egyptian Cat (286):" <<
          std::endl;
      std::cerr << "Max index: " << max_index << ", Max value: " << max_value
          << std::endl;
      return -1;
    }
    if (!is_close(max_value, expected_value)){
        std::cerr << "Invalid output value for expected Egyptian Cat ("<<expected_value<<"): "<< max_value <<
      std::endl;
      return -1;
    }
  }

  {
    constexpr float expected_index = 233; // Border collie
    constexpr float expected_value = 0.710397;
    auto net = build_mobilenet(h, w, number_of_workers);
    if (net.input_rows() != 224 || net.input_columns() != 224 || net.input_components() != channels) {
        std::cerr << "Invalid input shape for mobilenet 2: "<< net.input_rows() <<
        "x" << net.input_columns() << "x" << net.input_components() <<std::endl;
        return -1;
    }
    if (net.input().size() != 150528) {
        std::cerr << "Invalid input size for mobilenet 2: " << net.input().size()
        << std::endl;
        return -1;
    }
    if (net.output_rows() != 1 || net.output_columns() != 1 || net.output_components() != categories) {
        std::cerr << "Invalid output shape for mobilenet 2: " << net.output_rows() <<
        "x" << net.output_columns() <<std::endl;
        return -1;
    }
    std::copy(std::begin(input_dog), std::end(input_dog), net.input().begin());
    auto output = net();
    if (output.size() != categories) {
        std::cerr << "Invalid output size for mobilenet 2: " << output.size()
        << std::endl;
        return -1;
    }
    auto max_addr = std::max_element(output.begin(), output.end());
    float max_value = *max_addr;
    int max_index = max_addr - output.begin();
    if (max_index != expected_index){
      std::cerr << "Invalid output for expected Border collie (233):" <<
          std::endl;
      std::cerr << "Max index: " << max_index << ", Max value: " << max_value <<
          std::endl;
      return -1;
    }
      if (!is_close(max_value, expected_value)){
        std::cerr << "Invalid output value for expected Border collie ("<<expected_value<<"): "<< max_value <<
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
