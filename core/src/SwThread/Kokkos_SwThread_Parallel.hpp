//
// Created by Ryanxiejh on 2021/2/5.
//

#ifndef KOKKOS_KOKKOS_SWTHREAD_PARALLEL_HPP
#define KOKKOS_KOKKOS_SWTHREAD_PARALLEL_HPP

#if defined(KOKKOS_ENABLE_SWTHREAD)
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <KokkosExp_MDRangePolicy.hpp>

//----------------------------------------------------------------------------

namespace Kokkos{
namespace Impl{

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelFor Kokkos::SwThread with RangePolicy */
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                  Kokkos::SwThread> {
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;

 public:
  inline void execute() const {

//    //set range for athread
//    rp_range[0] = (this->m_policy).begin();
//    rp_range[1] = (this->m_policy).end();
//
//    //set execute pattern and policy
//    exec_patten = sw_Parallel_For;
//    target_policy = sw_Range_Policy;
//
//    //execution start
//    sw_create_threads();
//
//    //move the user function ptr to the next
//    user_func_index+=1;
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};


}
}

#endif //#if defined(KOKKOS_ENABLE_SWTHREAD)
#endif //KOKKOS_KOKKOS_SWTHREAD_PARALLEL_HPP
