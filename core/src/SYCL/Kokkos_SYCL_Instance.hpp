//
// Created by Ryanxiejh on 2021/2/17.
//

#ifndef KOKKOS_KOKKOS_SYCL_INSTANCE_HPP
#define KOKKOS_KOKKOS_SYCL_INSTANCE_HPP

#if defined(KOKKOS_ENABLE_SYCL)

#include <memory>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <CL/sycl.hpp>
#include <Kokkos_Concepts.hpp>
#include <SYCL/Kokkos_SYCL.hpp>
#include <impl/Kokkos_Error.hpp>

namespace Kokkos{
namespace Impl{

class SYCLInternal {
public:
  using size_type = size_t;

  SYCLInternal() = default;
  ~SYCLInternal();

  SYCLInternal(const SYCLInternal&) = delete;
  SYCLInternal& operator=(const SYCLInternal&) = delete;
  SYCLInternal& operator=(SYCLInternal&&) = delete;
  SYCLInternal(SYCLInternal&&)            = delete;

  static int was_initialized;
  static int was_finalized;

  static SYCLInternal& singleton();

  void initialize(const sycl::device& device);
  int verify_is_initialized(const char* const label) const;
  KOKKOS_INLINE_FUNCTION int is_initialized() const { return was_initialized != 0; }

  void finalize();

public:
    sycl::device m_device;
    int m_syclDev = 0;
    size_type* m_scratchSpace = nullptr;
    size_type* m_scratchFlags = nullptr;
    std::unique_ptr<sycl::queue> m_queue;
    //sycl::queue m_queue;
};

}  //namespace Impl
}  //namespace Kokkos

#endif //#if defined(KOKKOS_ENABLE_SYCL)
#endif //KOKKOS_KOKKOS_SYCL_INSTANCE_HPP
