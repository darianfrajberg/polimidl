#ifndef POLIMIDL_INTERNAL_NETWORK_HPP
#define POLIMIDL_INTERNAL_NETWORK_HPP

#include <algorithm>
#include <sstream>

#include "./aligned_buffer.hpp"
#include "../layers/alignment.hpp"
#include "./runner.hpp"
#include "./thread_pool.hpp"

namespace polimidl {
namespace internal {
template <std::size_t byte_alignment, typename layer_t>
std::size_t size_of(std::size_t input_rows, std::size_t input_columns,
               unsigned int number_of_workers) {
  std::size_t input_size = input_size_of<layer_t>(input_rows, input_columns);
  std::size_t output_size = output_size_of<layer_t>(
      layer_t::output_rows(input_rows), layer_t::output_columns(input_columns));
  std::size_t temporary_size = layer_t::temporary_size(input_rows,
                                                       input_columns,
                                                       number_of_workers);

  // The layers compute the size as number of cells, we need to convert them
  // into bytes, because the buffer will be allocated using a size in bytes.
  input_size *= sizeof(typename layer_t::type_t);
  output_size *= sizeof(typename layer_t::type_t);
  temporary_size *= sizeof(typename layer_t::type_t);

  // Round up the input size to a multiple of the alignment, this is necessary
  // to guaranteed the alignment of all the buffers involved.
  if (input_size % byte_alignment) {
    input_size += byte_alignment - input_size % byte_alignment;
  }
  // Round up the output size to a multiple of the alignment, this is necessary
  // to guaranteed the alignment of all the buffers involved.
  if (output_size % byte_alignment) {
    output_size += byte_alignment - output_size % byte_alignment;
  }
  // Round up the temporary size to a multiple of the alignment, this is necessary
  // to guaranteed the alignment of all the buffers involved.
  // It is also actually rounded to a multiple of alignment * number_of_workers
  // to guarantee the requested size per worker after the temporary span is
  // splitted.
  if (temporary_size % (byte_alignment * number_of_workers)) {
    temporary_size += (byte_alignment * number_of_workers) -
                      temporary_size % (byte_alignment * number_of_workers);
  }

  if constexpr (layer_t::is_in_place) {
    // If the layer is in place the input and the output share the same memory.
    return std::max(input_size, output_size) + temporary_size;
  }
  return input_size + output_size + temporary_size;
}

template <std::size_t byte_alignment, typename layer_t, typename next_layer_t,
          typename... Layers>
std::size_t size_of(std::size_t input_rows, std::size_t input_columns,
               unsigned int number_of_workers) {
  // Compute the size of the current layer
  std::size_t current_size = size_of<byte_alignment, layer_t>(
      input_rows, input_columns, number_of_workers);
  // Compute the size of the remaining layers
  std::size_t remaining_size = size_of<byte_alignment, next_layer_t, Layers...>(
    // The input size of the next layer is the output size of the current one.
    layer_t::output_rows(input_rows),
    layer_t::output_columns(input_columns),
    number_of_workers);
  // The requried memory is the one of the largest layer.
  return std::max(current_size, remaining_size);
}

template <typename type_t, typename... Layers>
class network {
public:
    network(std::size_t input_rows, std::size_t input_columns,
            unsigned int number_of_workers, Layers&&... layers)
      : // Initialize the aligned memory buffer.
        buffer_(size_of<alignment_t::byte_alignment, Layers...>(
            input_rows, input_columns, number_of_workers)),
        // Initialize the network runner.
        runner_(buffer_.template as_span<type_t>(), input_rows, input_columns,
                number_of_workers, std::forward<Layers>(layers)...),
        // Initialize the thread pool with the requested number of workers.
        thread_pool_(number_of_workers),
        // Statistics are not enabled by default.
        are_statistics_enabled_(false),
        accumulated_execution_duration_(0),
        accumulated_executions_(0) {
      // Run the optimization at initialization time.
      runner_.optimize_for(scheduler(&thread_pool_));
    }

    std::size_t input_rows() const { return runner_.input_rows(); }
    std::size_t input_columns() const { return runner_.input_columns(); }
    constexpr std::size_t input_components() const {
      return runner_.input_components();
    }
    std::size_t output_rows() const { return runner_.output_rows(); }
    std::size_t output_columns() const { return runner_.output_columns(); }
    constexpr std::size_t output_components() const {
      return runner_.output_components();
    }

    std::size_t used_memory() const {
      return buffer_.allocated_size();
    }

    span<type_t> input() const {
      return runner_.input();
    }

    span<type_t> run(){
      using std::chrono::high_resolution_clock;
      if (are_statistics_enabled_) {
        auto start = high_resolution_clock::now();
        runner_.run_with_statistics(scheduler(&thread_pool_));
        accumulated_execution_duration_ += high_resolution_clock::now() - start;
        ++accumulated_executions_;
      } else {
        runner_.run(scheduler(&thread_pool_));
      }
      return runner_.output();
    }

    span<type_t> operator()() { return run(); }

    bool are_statistics_enabled() const {
      return are_statistics_enabled_;
    }

    void enable_statistics() {
      are_statistics_enabled_ = true;
    }

    void disable_statistics() {
      are_statistics_enabled_ = false;
    }

    auto statistics(bool with_layers = false) {
      std::stringstream stream;
      if (accumulated_executions_ == 0) {
        stream << "No exections" << std::endl;
      } else {
        stream << "Execution Statistics" << std::endl;
        stream << "Number of executions = " << accumulated_executions_ <<
            std::endl;
        if (with_layers) {
          runner_.print_statistics(stream, 0, accumulated_executions_);
        }
        stream << "Network Avg Time = " <<
            std::chrono::duration_cast<std::chrono::microseconds>(
                accumulated_execution_duration_ / accumulated_executions_
            ).count() << " microseconds" << std::endl;
      }
      return stream.str();
    }

 private:
  using alignment_t = layers::buffer_alignment;
  aligned_buffer<alignment_t::byte_alignment> buffer_;
  runner<alignment_t::byte_alignment, type_t, false, Layers...> runner_;
  thread_pool thread_pool_;
  bool are_statistics_enabled_;
  std::chrono::high_resolution_clock::duration accumulated_execution_duration_;
  std::size_t accumulated_executions_;
};
}
}

#endif // POLIMIDL_INTERNAL_NETWORK_HPP
