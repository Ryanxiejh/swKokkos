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

  using traits = PolicyTraits<Properties...>;

  const typename traits::execution_space& space() const {
    static typename traits::execution_space m_space;
    return m_space;
  }

  template <class ExecSpace, class... OtherProperties>
  friend class TeamPolicyInternal;

  template <class... OtherProperties>
  TeamPolicyInternal(
      const TeamPolicyInternal<Kokkos::Threads, OtherProperties...>& p) {
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
      m_team_size = team_size_request > max_team_size ? max_team_size : team_size_request;
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
      m_team_size = team_size_request > max_team_size ? max_team_size : team_size_request;
  }

};


template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::SwThread> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Threads, Properties...>;
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

}
}

#endif //#if defined(KOKKOS_ENABLE_SWTHREAD)
#endif //KOKKOS_KOKKOS_SWTHREAD_PARALLEL_HPP
