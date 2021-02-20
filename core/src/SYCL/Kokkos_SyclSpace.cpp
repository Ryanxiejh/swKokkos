//
// Created by Ryanxiejh on 2021/2/19.
//
#include <SYCL/Kokkos_SyclSpace.hpp>

namespace Kokkos{

SyclSpace::SyclSpace() : m_device(SYCL().sycl_device()) {}

void* SyclSpace::allocate(const size_t arg_alloc_size) const {

  const sycl::queue& queue = *(SYCL().impl_internal_space_instance()->m_queue);

  void* const m_Ptr = sycl::malloc_device(arg_alloc_size, queue);

  queue.wait();

  if (m_Ptr == nullptr){
    const std::string msg(
        "SYCL usm allocate ERROR");
    Kokkos::Impl::throw_runtime_exception(msg);
  };

  return m_Ptr;
}

void SyclSpace::impl_deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const {

  const sycl::queue queue{SYCL().impl_internal_space_instance().m_device};

  sycl::free(arg_alloc_ptr, queue);
}

namespace Impl{
/*--------------------------------------------------------------------------*/
namespace {
void USM_memcpy(Kokkos::Impl::SYCLInternal& space, void* dst, const void* src, size_t n) {
  (void)USM_memcpy(*(space.m_queue), dst, src, n);
  queue.wait();
}

void USM_memcpy(void* dst, const void* src, size_t n) {
  Kokkos::Impl::SYCLInternal::singleton().m_queue->wait();
  const sycl::queue& queue = *(SYCL().impl_internal_space_instance()->m_queue);
  auto event = queue.memcpy(dst, src, n);
  event.wait();
}
}  // namespace

DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, Kokkos::SYCL>::DeepCopy(
        const Kokkos::SYCL& instance, void* dst, const void* src, size_t n) {
  USM_memcpy(*(instance.impl_internal_space_instance()), dst, src, n);
}

DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, Kokkos::SYCL>::DeepCopy(
        void* dst, const void* src, size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, Kokkos::SYCL>::DeepCopy(
        const Kokkos::SYCL& instance, void* dst, const void* src, size_t n) {
  USM_memcpy(*(instance.impl_internal_space_instance()), dst, src, n);
}

DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, Kokkos::SYCL>::DeepCopy(
        void* dst, const void* src, size_t n) {
  USM_memcpy(dst, src, n);
}

DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, Kokkos::SYCL>::DeepCopy(
        const Kokkos::SYCL& instance, void* dst, const void* src, size_t n) {
  USM_memcpy(*(instance.impl_internal_space_instance()), dst, src, n);
}

DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, Kokkos::SYCL>::DeepCopy(
        void* dst, const void* src, size_t n) {
  USM_memcpy(dst, src, n);
}

template <class ExecutionSpace>
DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, ExecutionSpace>::DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, Kokkos::SYCL>(dst, src, n);
}

template <class ExecutionSpace>
DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, ExecutionSpace>::DeepCopy(
        const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, Kokkos::SYCL>(Kokkos::SYCL(), dst, src, n);
    Kokkos::SYCL().fence();
}

template <class ExecutionSpace>
DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, ExecutionSpace>::DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, Kokkos::SYCL>(dst, src, n);
}

template <class ExecutionSpace>
DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, ExecutionSpace>::DeepCopy(
        const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, Kokkos::SYCL>(Kokkos::SYCL(), dst, src, n);
    Kokkos::SYCL().fence();
}

template <class ExecutionSpace>
DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, ExecutionSpace>::DeepCopy(void* dst, const void* src, size_t n) {
    (void)DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, Kokkos::SYCL>(dst, src, n);
}

template <class ExecutionSpace>
DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, ExecutionSpace>::DeepCopy(
        const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, Kokkos::SYCL>(Kokkos::SYCL(), dst, src, n);
    Kokkos::SYCL().fence();
}
/*--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------*/
#ifdef KOKKOS_DEBUG
SharedAllocationRecord<void, void> SharedAllocationRecord<
    Kokkos::SyclSpace, void>::s_root_record;
#endif

SharedAllocationRecord<Kokkos::SyclSpace, void>::
    SharedAllocationRecord(
        const Kokkos::SyclSpace& arg_space,
        const std::string& arg_label, const size_t arg_alloc_size,
        const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
          &SharedAllocationRecord<Kokkos::SyclSpace,
                                  void>::s_root_record,
#endif
          Kokkos::Impl::checked_allocation_with_header(arg_space, arg_label, arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_size, arg_dealloc),
      m_space(arg_space) {

#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::make_space_handle(arg_space.name()), arg_label, data(),
        arg_size);
  }
#endif
  SharedAllocationHeader header;

  // Fill in the Header information
  header.m_record = static_cast<SharedAllocationRecord<void, void>*>(this);

  strncpy(header.m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

  // Copy to device memory
  Kokkos::Impl::DeepCopy<Kokkos::SyclSpace, HostSpace>(
      RecordBase::m_alloc_ptr, &header, sizeof(SharedAllocationHeader));
}

std::string SharedAllocationRecord<Kokkos::SyclSpace, void>::get_label() const {

  SharedAllocationHeader header;

  Kokkos::Impl::DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace>(
      &header, RecordBase::head(), sizeof(SharedAllocationHeader));

  return std::string(header.m_label);
}

SharedAllocationRecord<Kokkos::SyclSpace, void>
        *SharedAllocationRecord<Kokkos::SyclSpace, void>::allocate(
                const Kokkos::SyclSpace& arg_space, const std::string& arg_label,
                const size_t arg_alloc_size) {
  return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
}

void SharedAllocationRecord<Kokkos::SyclSpace, void>::deallocate(
        SharedAllocationRecord<void, void>* arg_rec) {
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord<Kokkos::SyclSpace,
                       void>::~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    SharedAllocationHeader header;
    Kokkos::Impl::DeepCopy<Kokkos::SyclSpace,Kokkos::HostSpace>(
            &header, RecordBase::m_alloc_ptr, sizeof(SharedAllocationHeader));

    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::make_space_handle(
            Kokkos::SyclSpace::name()), header.m_label, data(), size());
  }
#endif

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

void* SharedAllocationRecord<Kokkos::SyclSpace, void>::allocate_tracked(
        const Kokkos::SyclSpace& arg_space, const std::string& arg_alloc_label,
        const size_t arg_alloc_size) {
  if (!arg_alloc_size) return (void *)0;

  SharedAllocationRecord* const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::SyclSpace,void>::deallocate_tracked(
        void* const arg_alloc_ptr) {
  if (arg_alloc_ptr != 0) {
    SharedAllocationRecord* const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void* SharedAllocationRecord<Kokkos::SyclSpace, void>::reallocate_tracked(
        void* const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord* const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord* const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<Kokkos::SyclSpace,Kokkos::SyclSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

SharedAllocationRecord<Kokkos::SyclSpace, void>*
SharedAllocationRecord<Kokkos::SyclSpace, void>::get_record(void* alloc_ptr) {
  using RecordSYCL = SharedAllocationRecord<Kokkos::SyclSpace, void>;
  using Header = SharedAllocationHeader;

  // Copy the header from the allocation
  Header head;

  Header const* const head_sycl =
      alloc_ptr ? Header::get_header(alloc_ptr) : (Header *)0;

  if (alloc_ptr) {
    Kokkos::Impl::DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace>(
        &head, head_sycl, sizeof(SharedAllocationHeader));
  }

  RecordSYCL* const record =
      alloc_ptr ? static_cast<RecordSYCL*>(head.m_record) : (RecordSYCL *)0;

  if (!alloc_ptr || record->m_alloc_ptr != head_sycl) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::SyclSpace , "
                    "void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord<Kokkos::SyclSpace, void>::print_records(
        std::ostream& s, const Kokkos::SyclSpace&, bool detail) {
  (void)s;
  (void)detail;
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord<void, void>* r = &s_root_record;

  char buffer[256];

  SharedAllocationHeader head;

  if (detail) {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<Kokkos::HostSpace,Kokkos::SyclSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));
      } else {
        head.m_label[0] = 0;
      }

      // Formatting dependent on sizeof(uintptr_t)
      const char* format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) {
        format_string =
            "SYCL addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx "
            "+ %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
        format_string =
            "SYCL addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ "
            "0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf(buffer, 256, format_string, reinterpret_cast<uintptr_t>(r),
               reinterpret_cast<uintptr_t>(r->m_prev),
               reinterpret_cast<uintptr_t>(r->m_next),
               reinterpret_cast<uintptr_t>(r->m_alloc_ptr), r->m_alloc_size,
               r->m_count, reinterpret_cast<uintptr_t>(r->m_dealloc),
               head.m_label);
      s << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  } else {
    do {
      if (r->m_alloc_ptr) {
        Kokkos::Impl::DeepCopy<Kokkos::HostSpace,Kokkos::SyclSpace>(
            &head, r->m_alloc_ptr, sizeof(SharedAllocationHeader));

        // Formatting dependent on sizeof(uintptr_t)
        const char* format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) {
          format_string = "SYCL [ 0x%.12lx + %ld ] %s\n";
        } else if (sizeof(uintptr_t) == sizeof(unsigned long long)) {
          format_string = "SYCL [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf(buffer, 256, format_string,
                 reinterpret_cast<uintptr_t>(r->data()), r->size(),
                 head.m_label);
      } else {
        snprintf(buffer, 256, "SYCL [ 0 + 0 ]\n");
      }
      s << buffer;
      r = r->m_next;
    } while (r != &s_root_record);
  }
#else
  Kokkos::Impl::throw_runtime_exception(
      "SharedAllocationHeader<SyclSpace>::print_records only works with "
      "KOKKOS_DEBUG enabled");
#endif
}
/*--------------------------------------------------------------------------*/

} //namespace Impl
} //namespace Kokkos
