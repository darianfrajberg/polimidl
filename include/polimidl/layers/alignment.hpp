#ifndef POLIMIDL_LAYERS_ALIGNMENT_HPP
#define POLIMIDL_LAYERS_ALIGNMENT_HPP

#include <Eigen/Dense>

namespace polimidl {
namespace layers {
class buffer_alignment {
 public:
  static constexpr std::size_t byte_alignment = 128; // 128 bytes
  static constexpr auto conditional_eigen_alignment(std::size_t alignment) {
      if (alignment % 16) {
          return Eigen::AlignmentType::Unaligned;
      }
      if (alignment % 32) {
          return Eigen::AlignmentType::Aligned16;
      }
      if (alignment % 64) {
          return Eigen::AlignmentType::Aligned32;
      }
      if (alignment % 128) {
          return Eigen::AlignmentType::Aligned64;
      }
      return Eigen::AlignmentType::Aligned128;
  }
};

template <typename type_t>
class alignment : public buffer_alignment {
 public:
  static constexpr std::size_t type_t_alignment =
      byte_alignment / sizeof(type_t);
  static constexpr auto eigen_alignment =
      buffer_alignment::conditional_eigen_alignment(byte_alignment);
  static constexpr auto conditional_eigen_alignment(std::size_t size) {
      size *= sizeof(type_t);
      return buffer_alignment::conditional_eigen_alignment(size * sizeof(type_t));
  }
};
}
}

#endif // POLIMIDL_LAYERS_ALIGNMENT_HPP
