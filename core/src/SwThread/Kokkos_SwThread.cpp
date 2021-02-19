//
// Created by Ryanxiejh on 2021/2/5.
//

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_SWTHREAD)

#include <Kokkos_SwThread.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos{

//void SwThread::impl_initialize(int thread_count){
//    if(thread_count <= 0) thread_count = 1;
//    num_threads =  thread_count;
//    sw_athread_init();
//}
//
//void SwThread::impl_finalize(){
//    sw_end_threads();
//}
//
//void SwThread::in_parallel(){
//    return num_threads;
//};
//
//void SwThread::concurrency(){
//    return num_threads;
//};

}


#endif