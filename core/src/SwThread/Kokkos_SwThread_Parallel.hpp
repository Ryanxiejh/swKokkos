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
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Parallel_Reduce.hpp>

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

    //set range for athread
    rp_range[0] = (this->m_policy).begin();
    rp_range[1] = (this->m_policy).end();

    //set execute pattern and policy
    exec_patten = sw_Parallel_For;
    target_policy = sw_Range_Policy;

    //execution start
    sw_create_threads();

    //move the user function ptr to the next
    user_func_index+=1;

  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

//----------------------------------------------------------------------------
/* ParallelFor Kokkos::SwThread with MDRangePolicy */
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::SwThread> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using WorkTag = typename MDRangePolicy::work_tag;

  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
                          // ).set_chunk_size(1) in ctor

 public:
  inline void execute() const {
    //set MDR host datas for athread
    sw_host_rank = (this->m_mdr_policy).rank;
    sw_host_tiles = (this->m_mdr_policy).m_num_tiles;
    for(int i = 0; i < sw_host_rank ; ++i) {
        sw_host_tile_nums[i] = ((this->m_mdr_policy).m_tile_end)[i];
        sw_host_tile[i] = ((this->m_mdr_policy).m_tile)[i];
        sw_host_lower[i] = ((this->m_mdr_policy).m_lower)[i];
        sw_host_upper[i] = ((this->m_mdr_policy).m_upper)[i];
    }

    //set execute pattern and policy
    exec_patten = sw_Parallel_For;
    target_policy = sw_MDR_Policy;

    //execution start
    sw_create_threads();

    //move the user function ptr to the next
    user_func_index+=1;
  }

  ParallelFor(const FunctorType &arg_functor, const MDRangePolicy &arg_policy)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};

//----------------------------------------------------------------------------
/* ParallelFor Kokkos::Threads with TeamPolicy */
class SwThreadTeamMember{
 private:
  int m_team_size;
  int m_team_rank;
  int m_league_size;
  int m_league_rank;

public:
    KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank; }
    KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
    KOKKOS_INLINE_FUNCTION int team_rank() const { return m_team_rank; }
    KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }

    SwThreadTeamMember()
        :m_team_size(0),
         m_team_rank(0),
         m_league_size(0),
         m_league_rank(0){}
};

template <class... Properties>
class TeamPolicyInternal<Kokkos::SwThread, Properties...>
    : public PolicyTraits<Properties...> {
 private:
  int m_league_size;
  int m_team_size;

 public:
  //! Tag this class as a kokkos execution policy
  //! Tag this class as a kokkos execution policy
  using execution_policy = TeamPolicyInternal;

  using member_type = Impl::SwThreadTeamMember;

  using traits = PolicyTraits<Properties...>;

  const typename traits::execution_space& space() const {
    static typename traits::execution_space m_space;
    return m_space;
  }

  template <class ExecSpace, class... OtherProperties>
  friend class TeamPolicyInternal;

  template <class... OtherProperties>
  TeamPolicyInternal(
      const TeamPolicyInternal<Kokkos::SwThread, OtherProperties...>& p) {
    m_league_size            = p.m_league_size;
    m_team_size              = p.m_team_size;
  }

  //----------------------------------------

  inline int team_size() const { return m_team_size; }
  inline int league_size() const { return m_league_size; }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     int vector_length_request = 1)
      : m_league_size(0),
        m_team_size(0) {
      //set leagea and team size
      const int max_team_size = num_threads;
      m_league_size = league_size_request;
      m_team_size = team_size_request > max_team_size ? max_team_size : team_size_request;
  }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     int /* vector_length_request */ = 1)
      : m_league_size(0),
        m_team_size(0) {
      //set leagea and team size
      const int max_team_size = num_threads;
      m_league_size = league_size_request;
      m_team_size = 1;
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_league_size(0),
        m_team_size(0) {
      //set leagea and team size
      const int max_team_size = num_threads;
      m_league_size = league_size_request;
      m_team_size = team_size_request > max_team_size ? max_team_size : team_size_request;
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     int /* vector_length_request */ = 1)
      : m_league_size(0),
        m_team_size(0) {
      //set leagea and team size
      const int max_team_size = num_threads;
      m_league_size = league_size_request;
      m_team_size = 1;
  }

};

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::SwThread> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::SwThread, Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;
  //const int m_shared;

 public:
  inline void execute() const {

    //set leagea and team size for athread
    sw_host_league_size = m_policy.league_size();
    sw_host_team_size = m_policy.team_size();

    //set execute pattern and policy
    exec_patten = sw_Parallel_For;
    target_policy = sw_TeamPolicy;

    //execution start
    sw_create_threads();

    //move the user function ptr to the next
    user_func_index+=1;
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy) {}
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::Threads and RangePolicy */

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::SwThread> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueTraits =
      Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename ValueTraits::pointer_type;
  using reference_type = typename ValueTraits::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const {

    //set reduce information for athread

    printf("//-----------------------------------------------\n");

    ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
            (void*)m_result_ptr);

    //如果使用默认reducer，即用户在构造函数中传进来reducer的类型时标量或者view，此时ReducerType是InvalidType，而这个情况下使默认使用
    //built-in reducer中的sum，所以在此进行特殊处理
    if(std::is_same<ReducerType,InvalidType>::value){
        is_buildin_reducer = 1;
        sw_reducer_type = sw_Reduce_SUM;

        printf("SwThread use default reducer!\n");
    }

    //如果使用的是built-in reducer，在built-in reducer的构造函数里会将is_buildin_reducer设为1
    //sw_reducer_type也会设置为相应值，这里只需获取其数据类型
    if(is_buildin_reducer == 1){
        if(std::is_same<typename ReducerTypeFwd::value_type,int>::value) sw_reducer_return_value_type = sw_TYPE_INT;
        else if(std::is_same<typename ReducerTypeFwd::value_type,long>::value) sw_reducer_return_value_type = sw_TYPE_LONG;
        else if(std::is_same<typename ReducerTypeFwd::value_type,float>::value) sw_reducer_return_value_type = sw_TYPE_FLOAT;
        else if(std::is_same<typename ReducerTypeFwd::value_type,double>::value) sw_reducer_return_value_type = sw_TYPE_DOUBLE;
        else if(std::is_same<typename ReducerTypeFwd::value_type,unsigned int>::value) sw_reducer_return_value_type = sw_TYPE_UINT;
        else if(std::is_same<typename ReducerTypeFwd::value_type,unsigned long>::value) sw_reducer_return_value_type = sw_TYPE_ULONG;
        else if(std::is_same<typename ReducerTypeFwd::value_type,char>::value) sw_reducer_return_value_type = sw_TYPE_CHAR;
        else if(std::is_same<typename ReducerTypeFwd::value_type,short>::value) sw_reducer_return_value_type = sw_TYPE_SHORT;
        else if(std::is_same<typename ReducerTypeFwd::value_type,unsigned short>::value) sw_reducer_return_value_type = sw_TYPE_USHORT;
        printf("SwThread use built-in reducer!\n");
    }
    //如果是custom reducer，则不作处理
    else printf("SwThread use custom reducer!\n");

    //获取reducer的数据长度
    sw_redecer_length = ValueTraits::value_count(
                            ReducerConditional::select(m_functor, m_reducer));

    printf("SwThread reducer length: %d\n",sw_redecer_length);
    printf("//-----------------------------------------------\n");

    //设置reducer的数据指针
    sw_reducer_ptr = m_result_ptr;

    //set range for athread
    rp_range[0] = (this->m_policy).begin();
    rp_range[1] = (this->m_policy).end();

    //set execute pattern and policy
    exec_patten = sw_Parallel_Reduce;
    target_policy = sw_Range_Policy;

    //execution start
    sw_create_threads();

    //move the user function ptr to the next
    user_func_index+=1;
    is_buildin_reducer=0; //重置该值
  }

  template <class HostViewType>
  ParallelReduce(
      const FunctorType &arg_functor, const Policy &arg_policy,
      const HostViewType &arg_result_view,
      typename std::enable_if<Kokkos::is_view<HostViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<HostViewType>::value,
                  "Kokkos::Threads reduce result must be a View");

    static_assert(
        std::is_same<typename HostViewType::memory_space, HostSpace>::value,
        "Kokkos::Threads reduce result must be a View in HostSpace");
  }

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

}
}

#endif //#if defined(KOKKOS_ENABLE_SWTHREAD)
#endif //KOKKOS_KOKKOS_SWTHREAD_PARALLEL_HPP
