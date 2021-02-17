//
// Created by Ryanxiejh on 2021/2/17.
//

#ifndef KOKKOS_KOKKOS_SYCLSPACE_HPP
#define KOKKOS_KOKKOS_SYCLSPACE_HPP

#if defined(KOKKOS_ENABLE_SYCL)

namespace Kokkos{

class SyclSpace{
public:
  using execution_space = SYCL;
  using memory_space    = SyclSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = Impl::SYCLInternal::size_type;
};

}

#endif
#endif //KOKKOS_KOKKOS_SYCLSPACE_HPP
