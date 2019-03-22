#include <chrono>
#include <iostream>

#include <polimidl/network.hpp>

#include "../examples/mobilenet_inputs/input_egyptian_cat.h" // normalized input 224x224x3 of egyptian cat
#include "../examples/mobilenet.hpp"

void test(unsigned int number_of_workers){
  constexpr int h = 224, w = 224;
  constexpr std::size_t categories = 1001;
  constexpr std::size_t ITERATIONS = 100;

  auto net = build_mobilenet(h, w, number_of_workers);
  std::cout << "Network size: " << net.used_memory() << std::endl;
  std::this_thread::sleep_for (std::chrono::seconds(1));
  std::copy(std::begin(input_egyptian_cat), std::end(input_egyptian_cat),
            net.input().begin());

  auto output = net();
  std::this_thread::sleep_for (std::chrono::seconds(1));
  net.enable_statistics();
  for (std::size_t iteration = 0; iteration < ITERATIONS; ++iteration){
      std::copy(std::begin(input_egyptian_cat), std::end(input_egyptian_cat),
                net.input().begin());
      output = net();
  }
  std::cout << "Run completed" << std::endl;
  auto max_addr = std::max_element(output.begin(), output.end());
  float max_value = *max_addr;
  int max_index = max_addr - output.begin();
  std::cout << "Max index: " << max_index << ", Max value: " << max_value
      << std::endl;
  std::cout << net.statistics(true) << std::endl;
}

int main(){
  using polimidl::max_number_of_workers;
  for (unsigned int number_of_workers = 1;
       number_of_workers <= max_number_of_workers(); ++number_of_workers) {
    std::cout << number_of_workers << " Worker/s" << std::endl;
    test(number_of_workers);
    std::cout << "--------------------" << std::endl;
  }
  return 0;
}
