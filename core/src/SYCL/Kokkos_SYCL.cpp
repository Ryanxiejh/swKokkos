//
// Created by Ryanxiejh on 2021/2/19.
//
#if defined(KOKKOS_ENABLE_SYCL)
#include <SYCL/Kokkos_SYCL.hpp>

namespace Kokkos{

SYCL::SYCL() : m_space_instance(&Impl::SYCLInternal::singleton()) {
  Impl::SYCLInternal::singleton().verify_is_initialized(
      "SYCL instance constructor");
}

void SYCL::fence() const { m_space_instance->m_queue->wait(); }

void SYCL::impl_initialize() {
  sycl::device device{sycl::default_selector_v };
  Impl::SYCLInternal::singleton().initialize(device);
}

void SYCL::impl_finalize() { Impl::SYCLInternal::singleton().finalize(); }

int SYCL::impl_is_initialized() {
  return Impl::SYCLInternal::singleton().is_initialized();
}

int SYCL::sycl_device() const {
  return impl_internal_space_instance()->m_syclDev;
}

} //namespace Kokkos

#endif //#if defined(KOKKOS_ENABLE_SYCL)
