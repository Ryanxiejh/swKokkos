//
// Created by Ryanxiejh on 2021/2/4.
//

#ifndef KOKKOS_KOKKOS_SWTHREAD_COMMONBASE_HPP
#define KOKKOS_KOKKOS_SWTHREAD_COMMONBASE_HPP

#if defined(KOKKOS_ENABLE_SWTHREAD)

#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>

typedef enum sw_Thread_state{
	sw_Thread_Active=0,
	sw_Thread_Inactive,
	sw_Thread_Teminating
}sw_ThreadState;
extern volatile int threadStates[64];
extern volatile int threadReduceStates[64];
extern long addr[65];
extern volatile int curViewIndex; //记录view的index
extern void* data_ptr[64];
extern int num_threads;
extern int N;
extern int rp_range[2];
extern size_t* data_dimension[64];
extern void(*currentFunc[64])(int);
extern void(*pForFunc[64])(int);
//extern HashTable* ht;
extern void* volatile reducer;

typedef enum sw_Layout{
    sw_LAYOUT_RIGHT=0, //普通view
    sw_LAYOUT_STRIDE //subview
}sw_Layout;


//syn变量
extern volatile int syn_value;

//锁变量
extern volatile int lock_req; //全局锁号
extern volatile int lock_cur; //当前锁号
extern volatile int swKokkos_slave_lock_req[64]; //从核持有的锁号
extern volatile void* atomic_fetch_val[64]; //原子fetch_and_*操作的旧变量


typedef enum sw_Execution_Pattern{
    sw_Parallel_For=0,
    sw_Parallel_Reduce,
    sw_Parallel_Scan
}sw_ExecutionPattern;
typedef enum sw_Execution_Policy{
    sw_Range_Policy=0,
    sw_MDR_Policy,
    sw_TeamPolicy
}sw_ExecutionPolicy;
extern volatile sw_ExecutionPattern exec_patten;
extern volatile sw_ExecutionPolicy target_policy;
extern volatile int user_func_index;

extern volatile int team_size;
extern volatile int league_size;


typedef enum sw_Reduce_ValueType{
    sw_TYPE_INT=0,
    sw_TYPE_LONG,
    sw_TYPE_FLOAT,
    sw_TYPE_DOUBLE,
    sw_TYPE_UINT,
    sw_TYPE_ULONG,
    sw_TYPE_CHAR,
    sw_TYPE_SHORT,
    sw_TYPE_USHORT
}sw_ValueType;

typedef enum sw_ReducerType{
    sw_Reduce_SUM=0,
    sw_Reduce_MIN,
    sw_Reduce_MAX
}sw_ReducerType;
//如果是要在cpp中给这些赋值，则保留；如果是在线程初始化时赋值，就用下面在ldm上定义的
extern volatile int is_buildin_reducer;
extern volatile sw_ReducerType reducer_type;
extern volatile sw_ValueType reducer_return_value_type;
extern volatile int redecer_length;

//scan
extern volatile long h_update[65];
extern volatile long temp_update[65];

//MDR host
extern volatile int data[640000][64];
extern int sw_host_tile_nums[8]; //每个维度tile的数量
extern int sw_host_tile[8]; //每个维度tile的大小
extern int sw_host_lower[8]; //每个维度tile的下界
extern int sw_host_upper[8]; //每个维度tile的上界
extern int sw_host_rank; //使用的维度数
extern int sw_host_cnt;
extern int sw_host_tiles; //总的tile数量

#endif //#if defined(KOKKOS_ENABLE_SWTHREAD)
#endif //KOKKOS_KOKKOS_SWTHREAD_COMMONBASE_HPP
