//
// Created by Ryanxiejh on 2021/2/4.
//

#ifndef KOKKOS_KOKKOS_SWTHREAD_HPP
#define KOKKOS_KOKKOS_SWTHREAD_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_SWTHREAD)

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <iosfwd>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <impl/Kokkos_Tags.hpp>

extern "C"{
    #include <SwThread/Kokkos_SwThread_CommonBase.hpp>
}
#include <SwThread/Kokkos_SwThread_HostBase.hpp>



namespace Kokkos{

class SwThread {
 public:
    //! \name Type declarations that all Kokkos devices must provide.
    // @{

    //! Tag this class as an execution space:
    using execution_space = SwThread;
    //! The size_type alias best suited for this device.
    using size_type = HostSpace::size_type;
    //! This device's preferred memory space.
    using memory_space = HostSpace;
    //! This execution space preferred device_type
    using device_type = Kokkos::Device<execution_space, memory_space>;

    //! This device's preferred array layout.
    using array_layout = LayoutRight;

    /// \brief  Scratch memory space
    using scratch_memory_space = ScratchMemorySpace<Kokkos::SwThread>;

    //@}

   /// \brief True if and only if this method is being called in a
   ///   thread-parallel function.
   inline static int in_parallel(){
       //return threads num
       return num_threads;
   }

   static void sw_initialize(){
       sw_create_threads();
   }

   static void sw_finalize(){
       sw_wait_threads();
       sw_end_threads();
   }

};


}


#endif //#if defined(KOKKOS_ENABLE_SWTHREAD)
#endif //KOKKOS_KOKKOS_SWTHREAD_HPP
