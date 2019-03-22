#ifndef POLIMIDL_LAYERS_INTERNAL_BATCH_NORM_RELU_HPP
#define POLIMIDL_LAYERS_INTERNAL_BATCH_NORM_RELU_HPP

#include <chrono>
#include <limits>

#include <Eigen/Dense>

#include "../alignment.hpp"
#include "../layer.hpp"

namespace polimidl {
namespace layers {
namespace internal {
template <typename type_t, typename components>
class batch_norm_relu : public layer<type_t, components, true> {
 public:
  batch_norm_relu(const type_t* beta, const type_t* mean,
                  const type_t* fused_gamma_variance_epsilon,
                  type_t min, type_t max) :
    beta_(beta),
    mean_(mean),
    fused_gamma_variance_epsilon_(fused_gamma_variance_epsilon),
    min_(min),
    max_(max),
    batch_size_(1) {}

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void operator()(input_t input, temporary_t temporary, output_t output,
                  std::size_t rows, std::size_t columns,
                  const scheduler_t& scheduler) const {
    const std::size_t cells = rows * columns;

    for (std::size_t start_cell = 0; start_cell < cells; start_cell += batch_size_) {
      const std::size_t end_cell = std::min(cells, start_cell + batch_size_);
      scheduler.schedule([=] (std::size_t worker) {
        const std::size_t cells_in_this_batch = end_cell - start_cell;
        // Given the fact that the input and output size are identical, and the
        // layers is in place, they are guaranteed to be perfectly overlapping.
        auto i_data = &input[components::value * start_cell];
        array_data_t val(i_data, components::value, cells_in_this_batch);
        val = (((val.colwise() - mean_)
            .colwise() * fused_gamma_variance_epsilon_)
            .colwise() + beta_)
            .cwiseMax(type_t(min_)).cwiseMin(type_t(max_));
      });
    }
    scheduler.wait();
  }

  template <typename input_t, typename temporary_t, typename output_t,
            typename scheduler_t>
  void optimize_for(input_t input, temporary_t temporary, output_t output,
                    std::size_t rows, std::size_t columns, const scheduler_t& scheduler) {
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
      // Ensure the alignment of the Eigen Array.
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
  using array_data_t = Eigen::Map<
      Eigen::Array<type_t, components::value, Eigen::Dynamic>,
      alignment_t::eigen_alignment>;
  using array_coefficients_t = Eigen::Map<
      const Eigen::Array<type_t, components::value, 1>,
      alignment_t::eigen_alignment>;
  array_coefficients_t beta_;
  array_coefficients_t mean_;
  array_coefficients_t fused_gamma_variance_epsilon_;
  const type_t min_;
  const type_t max_;
  std::size_t batch_size_;
};
}
}
}

#endif // POLIMIDL_LAYERS_INTERNAL_BATCH_NORM_RELU_HPP
