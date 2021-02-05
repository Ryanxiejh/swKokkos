//
// Created by Ryanxiejh on 2021/2/4.
//

#ifndef KOKKOS_KOKKOS_SWTHREAD_HOSTBASE_HPP
#define KOKKOS_KOKKOS_SWTHREAD_HOSTBASE_HPP
#if defined(KOKKOS_ENABLE_SWTHREAD)

void sw_athread_init();

void sw_create_threads();

void sw_end_threads();

#endif //#if defined(KOKKOS_ENABLE_SWTHREAD)
#endif //KOKKOS_KOKKOS_SWTHREAD_HOSTBASE_HPP
