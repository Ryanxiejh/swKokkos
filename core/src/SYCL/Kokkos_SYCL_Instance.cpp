//
// Created by Ryanxiejh on 2021/2/19.
//
#include <SYCL/Kokkos_SYCL_Instance.hpp>

namespace Kokkos{
namespace Impl{

SYCLInternal::~SYCLInternal() {
  if (m_scratchSpace || m_scratchFlags) {
    std::cerr << "Kokkos::SYCL ERROR: Failed to call Kokkos::SYCL::finalize()"
              << std::endl;
    std::cerr.flush();
  }

  m_scratchSpace = nullptr;
  m_scratchFlags = nullptr;
}

int SYCLInternal::was_initialized = 0;
int SYCLInternal::was_finalized = 0;

SYCLInternal& SYCLInternal::singleton() {
  static SYCLInternal self;
  return self;
}

void SYCLInternal::initialize(const sycl::device& device) {
  if (was_finalized)
    Kokkos::abort("Calling SYCL::initialize after SYCL::finalize is illegal\n");

  if (is_initialized()) return;

  was_initialized = 1;

  if (!HostSpace::execution_space::impl_is_initialized()) {
    const std::string msg(
        "SYCL::initialize ERROR : HostSpace::execution_space is not "
        "initialized");
    Kokkos::Impl::throw_runtime_exception(msg);
  }

  const bool ok_init = nullptr == m_scratchSpace || nullptr == m_scratchFlags;
  const bool ok_dev  = true;
  if (ok_init && ok_dev) {
    m_queue = std::make_unique<sycl::queue>(device);
    m_device = device;
  } else {
    std::ostringstream msg;
    msg << "Kokkos::SYCL::initialize FAILED";

    if (!ok_init) {
      msg << " : Already initialized";
    }
    Kokkos::Impl::throw_runtime_exception(msg.str());
  }
}

int SYCLInternal::verify_is_initialized(const char* const label) const {
  if (!is_initialized()) {
    std::cerr << "Kokkos::SYCL::" << label << " : ERROR device not initialized"
              << std::endl;
  }
  return is_initialized();
}

void SYCLInternal::finalize() {
  //SYCL().fence();
  was_finalized = 1;
  if (nullptr != m_scratchSpace || nullptr != m_scratchFlags) {
    // FIXME_SYCL
    std::abort();
  }
  m_queue.reset();
}

}   // namespace Impl
}   // namespace Kokkos


