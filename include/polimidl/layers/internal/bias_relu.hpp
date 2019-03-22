#ifndef POLIMIDL_LAYERS_INTERNAL_BIAS_RELU_HPP
#define POLIMIDL_LAYERS_INTERNAL_BIAS_RELU_HPP

#include <algorithm>

#include "../alignment.hpp"
#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components>
class bias_relu : public layer<type_t, components, true> {
 public:
  bias_relu(const type_t* bias, type_t min, type_t max) :
      bias_(bias), min_(min), max_(max), batch_size_(1) {}

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    const std::size_t cells = rows * columns;

    for (std::size_t start_cell = 0; start_cell < cells;
        start_cell += batch_size_) {
      const std::size_t end_cell = std::min(cells, start_cell + batch_size_);
      scheduler.schedule([=] (std::size_t worker) {
        for (std::size_t cell = start_cell; cell < end_cell; ++cell) {
          std::transform(&input[components::value * cell],
                         &input[components::value * (cell + 1)],
                         bias_,
                         &output[components::value * cell],
                         [&](type_t value, type_t bias) {
            return std::clamp(value + bias, min_, max_);
          });
        }
      });
    }
    scheduler.wait();
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void optimize_for(input_t input, temporary_t temporary, output_t output,
                    std::size_t rows, std::size_t columns,
                    const scheduler_t& scheduler) {
    const std::size_t cells = rows * columns;
    if (scheduler.number_of_workers() == 1) {
      batch_size_ = cells;
      return;
    }

    auto best_duration = std::chrono::high_resolution_clock::duration::max();

    std::size_t best_batch_size = 1;
    auto test_and_maybe_set = [&]() {
      for (std::size_t iteration = 0; iteration < 3; ++iteration) {
        const auto start = std::chrono::high_resolution_clock::now();
        operator()(input, temporary, output, rows, columns, scheduler);
        const auto current_duration =
            std::chrono::high_resolution_clock::now() - start;
        if (current_duration < best_duration) {
          best_duration = current_duration;
          best_batch_size = batch_size_;
        }
      }
    };
    batch_size_ = 1;
    while (batch_size_ < cells) {
      if ((batch_size_ * components::value) % alignment_t::type_t_alignment) {
        batch_size_ *= 2;
        continue;
      }
      test_and_maybe_set();
      batch_size_ *= 2;
    }
    batch_size_ = cells;
    test_and_maybe_set();
    batch_size_ = best_batch_size;
  }

 private:
  using alignment_t = alignment<type_t>;
  const type_t* bias_;
  const type_t min_;
  const type_t max_;
  std::size_t batch_size_;
};
}
}
}

#endif  // POLIMIDL_LAYERS_INTERNAL_BIAS_RELU_HPP
