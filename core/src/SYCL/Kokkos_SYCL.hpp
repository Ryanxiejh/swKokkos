//
// Created by Ryanxiejh on 2021/2/17.
//

#ifndef KOKKOS_KOKKOS_SYCL_HPP
#define KOKKOS_KOKKOS_SYCL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_SYCL)

#include <Kokkos_Core_fwd.hpp>
#include <CL/sycl.hpp>
#include <iostream>
#include <SYCL/Kokkos_SyclSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

namespace Kokkos{

class SYCL{
public:
  using execution_space = SYCL;
  using memory_space    = SyclSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using array_layout = LayoutRight;
  using size_type    = memory_space::size_type;

  using scratch_memory_space = ScratchMemorySpace<SYCL>;

  ~SYCL() = default;
  SYCL();

  SYCL(SYCL&&)      = default;
  SYCL(const SYCL&) = default;
  SYCL& operator=(SYCL&&) = default;
  SYCL& operator=(const SYCL&) = default;

  KOKKOS_INLINE_FUNCTION static int in_parallel() {
//      cl::sycl::queue q;
//      int N = 5;
//      int *data = cl::sycl::malloc_shared<int>(N, q);
//      for(int i=0; i<N; i++) data[i] = i;
//
//      q.parallel_for(cl::sycl::range<1>(N), [=] (cl::sycl::id<1> i){
//        data[i] *= 2;
//      }).wait();
//
//      for(int i=0; i<N; i++) std::cout << data[i] << std::endl;
//      free(data, q);

      return 1;
  }

  //static void impl_static_fence();

  void fence() const;

  KOKKOS_INLINE_FUNCTION static size_t concurrency() {
      return SYCL().impl_internal_space_instance()->m_device.get_info<sycl::info::device::max_compute_units>();
  }

  KOKKOS_INLINE_FUNCTION static const char* name() { return "SYCL" ;};

  static void impl_initialize();

  static void impl_finalize();

  static int impl_is_initialized();

  KOKKOS_INLINE_FUNCTION Impl::SYCLInternal* impl_internal_space_instance() const {
    return m_space_instance;
  }

  int sycl_device() const;

  uint32_t impl_instance_id() const noexcept { return 0; }

private:
    Impl::SYCLInternal* m_space_instance;
};

namespace Profiling {
namespace Experimental {
template <>
struct DeviceTypeTraits<SYCL> {
  static constexpr DeviceType id = DeviceType::SYCL;
};
}  // namespace Experimental
}  // namespace Profiling
} //namespace Kokkos

#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <SYCL/Kokkos_SYCL_Parallel.hpp>
#endif //#if defined(KOKKOS_ENABLE_SYCL)
#endif //KOKKOS_KOKKOS_SYCL_HPP
