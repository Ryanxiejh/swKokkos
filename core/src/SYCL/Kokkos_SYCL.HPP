//
// Created by Ryanxiejh on 2021/2/17.
//

#ifndef KOKKOS_KOKKOS_SYCL_HPP
#define KOKKOS_KOKKOS_SYCL_HPP

#if defined(KOKKOS_ENABLE_SYCL)

#include <Kokkos_Macros.hpp>
#include <CL/sycl.hpp>
#include <iostream>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

namespace Kokkos{

class SYCL{

  using execution_space = SYCL;
  using memory_space    = HostSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using array_layout = LayoutLeft;
  using size_type    = memory_space::size_type;

  using scratch_memory_space = ScratchMemorySpace<SYCL>;

  ~SYCL() = default;
  SYCL();

  SYCL(SYCL&&)      = default;
  SYCL(const SYCL&) = default;
  SYCL& operator=(SYCL&&) = default;
  SYCL& operator=(const SYCL&) = default;

public:
  KOKKOS_INLINE_FUNCTION static int in_parallel() {
      cl::sycl::queue q;
      int N = 5;
      int *data = cl::sycl::malloc_shared<int>(N, q);
      for(int i=0; i<N; i++) data[i] = i;

      q.parallel_for(cl::sycl::range<1>(N), [=] (cl::sycl::id<1> i){
        data[i] *= 2;
      }).wait();

      for(int i=0; i<N; i++) std::cout << data[i] << std::endl;
      free(data, q);

      return 1;
  }

};
