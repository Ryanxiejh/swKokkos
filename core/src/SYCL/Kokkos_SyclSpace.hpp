//
// Created by Ryanxiejh on 2021/2/17.
//

#ifndef KOKKOS_KOKKOS_SYCLSPACE_HPP
#define KOKKOS_KOKKOS_SYCLSPACE_HPP

#if defined(KOKKOS_ENABLE_SYCL)
#include <Kokkos_Core_fwd.hpp>
#include <SYCL/Kokkos_SYCL.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Concepts.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <Kokkos_HostSpace.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>
#include <impl/Kokkos_MemorySpace.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

namespace Kokkos{

class SyclSpace{
public:
  using execution_space = SYCL;
  using memory_space    = SyclSpace;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = Impl::SYCLInternal::size_type;

  /**\brief  Default memory space instance */
  SyclSpace();
  SyclSpace(SyclSpace&& rhs)      = default;
  SyclSpace(const SyclSpace& rhs) = default;
  SyclSpace& operator=(SyclSpace&&) = default;
  SyclSpace& operator=(const SyclSpace&) = default;
  ~SyclSpace()                           = default;

  /**\brief  Allocate untracked memory in the space */
  void* allocate(const size_t arg_alloc_size) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return "SyclUSMSpace"; }

};

namespace Impl{
/*--------------------------------------------------------------------------*/
static_assert(Kokkos::Impl::MemorySpaceAccess<
                  Kokkos::SyclSpace,
                  Kokkos::SyclSpace>::assignable,
              "");

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::SyclSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::SyclSpace,
                         Kokkos::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
template <>
struct DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, Kokkos::SYCL> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Kokkos::SYCL&, void* dst, const void* src, size_t);
};

template <>
struct DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, Kokkos::SYCL> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Kokkos::SYCL&, void* dst, const void* src, size_t);
};

template <>
struct DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, Kokkos::SYCL> {
  DeepCopy(void* dst, const void* src, size_t);
  DeepCopy(const Kokkos::SYCL&, void* dst, const void* src, size_t);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::SyclSpace, Kokkos::SyclSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n);
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::SyclSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n);
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::SyclSpace, Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n);
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n);
};
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
template <>
class SharedAllocationRecord<Kokkos::SyclSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

#ifdef KOKKOS_DEBUG
  static RecordBase s_root_record;
#endif

  const Kokkos::SyclSpace m_space;

 protected:
  SharedAllocationRecord(const Kokkos::SyclSpace& arg_space,
                         const std::string& arg_label,
                         const size_t arg_alloc_size,
                         const RecordBase::function_type arg_dealloc = &deallocate);

  ~SharedAllocationRecord();

 public:
  std::string get_label() const;

  static SharedAllocationRecord* allocate(const Kokkos::SyclSpace& arg_space,
                                          const std::string& arg_label,
                                          const size_t arg_alloc_size);

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const Kokkos::SyclSpace& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  static void print_records(std::ostream&,
                            const Kokkos::SyclSpace&,
                            bool detail = false);
};
/*--------------------------------------------------------------------------*/
} //namespace Impl

} //namespace Kokkos

#endif //#if defined(KOKKOS_ENABLE_SYCL)
#endif //KOKKOS_KOKKOS_SYCLSPACE_HPP
