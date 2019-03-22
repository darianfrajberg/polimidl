#ifndef POLIMIDL_INTERNAL_ALIGNED_BUFFER_HPP
#define POLIMIDL_INTERNAL_ALIGNED_BUFFER_HPP

#include <cstdint>
#include <memory>

#include "./span.hpp"

namespace polimidl {
namespace internal {
template <std::size_t alignment>
class aligned_buffer {
 public:
  aligned_buffer(std::size_t size) :
      size_(size),
      buffer_(std::make_unique<std::int8_t[]>(size_ + alignment)),
      begin_(align_begin(buffer_.get(), size_)) {}

  auto size() const { return size_; }
  auto allocated_size() const { return size() + alignment; }

  template <typename type_t>
  auto as_span() const {
    return span<type_t>(reinterpret_cast<type_t*>(begin_),
                        size_ / sizeof(type_t));
  }

 private:
  static std::int8_t* align_begin(std::int8_t* begin, std::size_t size) {
    std::size_t space = size + alignment;
    void* aligned_begin = reinterpret_cast<void*>(begin);
    std::align(alignment, size, aligned_begin, space);
    return reinterpret_cast<std::int8_t*>(aligned_begin);
  }

  std::size_t size_;
  std::unique_ptr<std::int8_t[]> buffer_;
  std::int8_t* begin_;
};
}
}

#endif // POLIMIDL_INTERNAL_ALIGNED_BUFFER_HPP
