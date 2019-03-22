#ifndef POLIMIDL_INTERNAL_MEMORY_LAYOUT_HPP
#define POLIMIDL_INTERNAL_MEMORY_LAYOUT_HPP

#include <algorithm>

#include "./span.hpp"

namespace polimidl {
namespace internal {
template <std::size_t alignment, typename layer_t, bool is_inverted>
auto input_of(span<typename layer_t::type_t> buffer,
              std::size_t input_size) {
  // The begin and end of the buffer are guaranteed to be aligned.
  constexpr std::size_t type_t_alignment =
      alignment / sizeof(typename layer_t::type_t);
  using span_t = span<typename layer_t::type_t>;
  if constexpr (is_inverted) {
    // If the input is at the end of the buffer we should ensure its aligment.
    std::size_t offset = input_size % type_t_alignment;
    if (offset != 0) {
      // If the size of the input is not multiple of the alignment we need to
      // shift it fowards the beginning.
      return span_t(input_size, buffer.end() - type_t_alignment + offset);
    }
    // If the size of the input is multiple of the alignment it is already
    // aligned.
    return span_t(input_size, buffer.end());
  }
  // If the input is at the beginning of the buffer it is aligned.
  return span_t(buffer.begin(), input_size);
}

template <std::size_t alignment, typename layer_t, bool is_inverted>
auto output_of(span<typename layer_t::type_t> buffer,
              std::size_t output_size) {
  // The begin and end of the buffer are guaranteed to be aligned.
  constexpr std::size_t type_t_alignment =
      alignment / sizeof(typename layer_t::type_t);
  using span_t = span<typename layer_t::type_t>;
  if constexpr ((is_inverted && layer_t::is_in_place) ||
                (!is_inverted && !layer_t::is_in_place)) {
    // If the output is at the end of the buffer we should ensure its aligment.
    std::size_t offset = output_size % type_t_alignment;
    if (offset != 0) {
      // If the size of the output is not multiple of the alignment we need to
      // shift it fowards the beginning.
      return span_t(output_size, buffer.end() - type_t_alignment + offset);
    }
    // If the size of the output is multiple of the alignment it is already
    // aligned.
    return span_t(output_size, buffer.end());
  }
  // If the output is at the beginning of the buffer it is aligned.
  return span_t(buffer.begin(), output_size);
}

template <std::size_t alignment, typename layer_t, bool is_inverted>
auto temporary_of(span<typename layer_t::type_t> buffer, std::size_t input_size,
                  std::size_t output_size, unsigned int number_of_workers) {
  constexpr std::size_t type_t_alignment =
      alignment / sizeof(typename layer_t::type_t);
  using span_t = span<typename layer_t::type_t>;
  // We round the size of the input to a multiple of type_t_alignment to ensure
  // that the span returned by input_of is aligned.  
  if (input_size % type_t_alignment) {
    input_size += type_t_alignment - input_size % type_t_alignment;
  }
  auto input = input_of<alignment, layer_t, is_inverted>(buffer, input_size);
  // We round the size of the output to a multiple of type_t_alignment to ensure
  // that the span returned by output_of is aligned.
  if (output_size % type_t_alignment) {
    output_size += type_t_alignment - output_size % type_t_alignment;
  }
  auto output = output_of<alignment, layer_t, is_inverted>(buffer, output_size);
  std::size_t temporary_size = [=]() {
    if constexpr (layer_t::is_in_place) {
      // If the layer is in place input and output share the same memory.
      return buffer.size() - std::max(input.size(), output.size());
    }
    // If the layer is not in place input and output do not share the same
    // memory.
    return buffer.size() - input.size() - output.size();
  }();
  // We round the requested temporary size to
  // type_t_alignment * number_of_workers this guaranteed that the per thread
  // temporary buffers, once splitted from the main one, will be aligned.
  if (temporary_size % (type_t_alignment * number_of_workers)) {
    temporary_size -= temporary_size % (type_t_alignment * number_of_workers);
  }
  if constexpr (is_inverted) {
    if constexpr (layer_t::is_in_place) {
      // If the layer is inverted and in place the temporary is at the beginning
      // of the buffer.
      return span_t(buffer.begin(), temporary_size);
    }
    // If the the layer is inverted, but not in place the temporary is after the
    // end of the output, which is at the beginning of the buffer, and before
    // the beginning of the input, which is at the end of the buffer.
    return span_t(output.end(), temporary_size);
  }
  if constexpr (layer_t::is_in_place) {
    // If the layer is not inverted and in place the temporary after the end of
    // both input and output, which are overlapping at the beginning of the
    // buffer.
    return span_t(std::max(input.end(), output.end()), temporary_size);
  }
  // If the the layer is inverted, but not in place the temporary is after the
  // end of the input, which is at the beginning of the buffer, and before
  // the beginning of the output, which is at the end of the buffer.
  return span_t(input.end(), temporary_size);
}
}
}

#endif // POLIMIDL_INTERNAL_MEMORY_LAYOUT_HPP
