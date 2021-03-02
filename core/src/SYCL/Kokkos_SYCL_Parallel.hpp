//
// Created by Ryanxiejh on 2021/2/17.
//

#ifndef KOKKOS_KOKKOS_SYCL_PARALLEL_HPP
#define KOKKOS_KOKKOS_SYCL_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_SYCL)
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Core.hpp>
#include <stdlib.h>
#include <Kokkos_Parallel.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include <SYCL/Kokkos_SYCL_IterateTile.hpp>
#include <iostream>
#include <functional>

namespace Kokkos{
namespace Impl{

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelFor Kokkos::SYCL with RangePolicy */
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::SYCL> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 private:
  using Member       = typename Policy::member_type;
  using WorkTag      = typename Policy::work_tag;

  const FunctorType m_functor;
  const Policy m_policy;

 private:
  ParallelFor()        = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;

  template <typename Functor>
  static void sycl_direct_launch(const Policy& policy, const Functor& functor) {
    // Convenience references
    const Kokkos::SYCL& space = policy.space();
    Kokkos::Impl::SYCLInternal& instance = *space.impl_internal_space_instance();
    sycl::queue& q = *(instance.m_queue);

    q.wait();

    const typename Policy::index_type work_range = policy.end() - policy.begin();
    const typename Policy::index_type offset = policy.begin();

    q.submit([functor, work_range, offset](sycl::handler& cgh) {
      sycl::range<1> range(work_range);

      cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
        const typename Policy::index_type id =
            static_cast<typename Policy::index_type>(item.get_linear_id()) + offset;
        if constexpr (std::is_same<WorkTag, void>::value)
          functor(id);
        else
          functor(WorkTag(), id);
      });
    });

    q.wait();
  }

  //在usm中构造functor
  void sycl_indirect_launch() const {
    std::cout << "sycl_indirect_launch !!!" << std::endl;
    const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
    auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
    new (usm_functor_ptr) FunctorType(m_functor);
    sycl_direct_launch(m_policy,std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr))));
    sycl::free(usm_functor_ptr,queue);
  }

 public:

  void execute() const {
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_policy, m_functor);
    else
      sycl_indirect_launch();
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
/* ParallelFor Kokkos::SYCL with MDRangePolicy */
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, Kokkos::SYCL> {
 private:
  typedef Kokkos::MDRangePolicy<Traits...> MDRangePolicy;
  typedef typename MDRangePolicy::impl_range_policy Policy;

  typedef typename MDRangePolicy::work_tag WorkTag;

  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;

  typedef typename Kokkos::Impl::SyclIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>
      iterate_type;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
                          // ).set_chunk_size(1) in ctor

  template <typename Functor>
  /*static*/ void sycl_direct_launch(const Policy& policy, const Functor& functor) const{
    // Convenience references
    const Kokkos::SYCL& space = policy.space();
    Kokkos::Impl::SYCLInternal& instance = *space.impl_internal_space_instance();
    sycl::queue& q = *(instance.m_queue);

    q.wait();

    const typename Policy::index_type work_range = policy.end() - policy.begin();
    const typename Policy::index_type offset = policy.begin();

//    auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),q);
//    new (usm_functor_ptr) FunctorType(m_functor);
//    FunctorType& func = std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr)));

    MDRangePolicy mdr = m_mdr_policy;

    //std::cout << "work_range： " << work_range << std::endl;

    q.submit([=](sycl::handler& cgh) {
      sycl::range<1> range(work_range);
      sycl::stream out(1024, 256, cgh);
      cgh.parallel_for(range, [=](cl::sycl::item<1> item) {
        const typename Policy::index_type id =
            static_cast<typename Policy::index_type>(item.get_linear_id()) + offset;
         const iterate_type iter(mdr,functor);
         iter(id);
         //functor(id);
//         if(id==0){
//             out << "sycl kernel run id: " << id << sycl::endl;
//             out << (iter.m_func.a)(id,0) << " " << (iter.m_func.a)(id,1) << " " << (iter.m_func.a)(id,2) << sycl::endl;
//         }
      });
    });

    q.wait();
  }

  //在usm中构造functor
  void sycl_indirect_launch() const {
      //这种方法在gpu上跑会有问题，所有的数据都变为0，而不是目标值
//    std::cout << "sycl_indirect_launch !!!" << std::endl;
//    const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
//    auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
//    auto usm_iter_ptr = sycl::malloc_shared(sizeof(iterate_type),queue);
//    new (usm_functor_ptr) FunctorType(m_functor);
//    new (usm_iter_ptr) iterate_type(m_mdr_policy,*(static_cast<FunctorType*>(usm_functor_ptr)));
//    sycl_direct_launch(m_policy, std::reference_wrapper(*(static_cast<iterate_type*>(usm_iter_ptr))));
//    sycl::free(usm_functor_ptr,queue);
//    sycl::free(usm_iter_ptr,queue);

    std::cout << "sycl_indirect_launch !!!" << std::endl;
    const sycl::queue& queue = *(m_policy.space().impl_internal_space_instance()->m_queue);
    auto usm_functor_ptr = sycl::malloc_shared(sizeof(FunctorType),queue);
    new (usm_functor_ptr) FunctorType(m_functor);
    sycl_direct_launch(m_policy,std::reference_wrapper(*(static_cast<FunctorType*>(usm_functor_ptr))));
    sycl::free(usm_functor_ptr,queue);
  }

 public:
   void execute() const {
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_policy, m_functor);
    else
      sycl_indirect_launch();
  }

  ParallelFor(const FunctorType &arg_functor, const MDRangePolicy &arg_policy)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::SYCL and RangePolicy */
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::SYCL> {
 private:
  typedef Kokkos::RangePolicy<Traits...> Policy;

  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;

  typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                             FunctorType, ReducerType>
      ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type WorkTagFwd;

  typedef Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>
      ValueTraits;
  typedef Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd> ValueInit;

  typedef typename ValueTraits::pointer_type pointer_type;
  typedef typename ValueTraits::reference_type reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <typename T>
  struct ExtendedReferenceWrapper : std::reference_wrapper<T> {
    using std::reference_wrapper<T>::reference_wrapper;

    using value_type = typename FunctorValueTraits<T, WorkTag>::value_type;

    template <typename Dummy = T>
    std::enable_if_t<std::is_same_v<Dummy, T> &&
                     ReduceFunctorHasInit<Dummy>::value>
    init(value_type& old_value, const value_type& new_value) const {
      return this->get().init(old_value, new_value);
    }

    template <typename Dummy = T>
    std::enable_if_t<std::is_same_v<Dummy, T> &&
                     ReduceFunctorHasJoin<Dummy>::value>
    join(value_type& old_value, const value_type& new_value) const {
      return this->get().join(old_value, new_value);
    }

    template <typename Dummy = T>
    std::enable_if_t<std::is_same_v<Dummy, T> &&
                     ReduceFunctorHasFinal<Dummy>::value>
    final(value_type& old_value) const {
      return this->get().final(old_value);
    }
  };

 public:
  inline void execute() const {

  }

  template <class ViewType>
  ParallelReduce(
      const FunctorType &arg_functor, const Policy &arg_policy,
      const ViewType &arg_result_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  inline ParallelReduce(const FunctorType &arg_functor, Policy arg_policy,
                        const ReducerType &reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

} // namespace Kokkos
} // namespace Impl



#endif //#if defined(KOKKOS_ENABLE_SYCL)
#endif //KOKKOS_KOKKOS_SYCL_PARALLEL_HPP
