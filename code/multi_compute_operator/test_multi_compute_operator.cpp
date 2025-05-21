/**
 * @file test_multi_compute_operator.cpp
 * @author Ruichen Han (hanruichen@ruc.edu.cn)
 * @brief test multidimensional computation operator
 * @version 0.1
 * @date 2023-05-09
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <sched.h>    /* sched_setaffinity */
#include <stdio.h>    /* printf */
#include <sys/time.h> /* gettimeofday */
#include <getopt.h>   /* getopt */
#include <stdlib.h>   /* exit */
#include <string.h>   /* strcmp */
#include <limits.h>   /* INT_MAX */
#include <malloc.h>   /* malloc */
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <numaif.h>
#include "../include/gendata_util.hpp"
#include "../include/statistical_analysis_util.hpp"
#include "../include/metadata.h"
/**
 * @brief command line handling functions
 * 
 * @param progname 
 * @return void 
 */
void print_help(char *progname)
{
    printf("Usage: %s [options]\n", progname);

    printf("\
    join configuration options, with default values in [] :             \n\
       -n --nthreads=<N>  Number of threads to use <N> [2]                    \n\
       --SF               Size of dataset                                     \n\
       --d-sele           Selection rate of column on date table              \n\
       --d-bitmap         Flag of  bitmap index on date<1——bitmap 0——vector>  \n\
       --d-groups         Number of groups of columns on date                 \n\
       --s-sele           Selection rate of column on supplier table          \n\
       --s-bitmap         Flag of bitmap index on supplier<1——bitmap 0——vector>\n\
       --s-groups         Number of groups of columns on supplier             \n\
       --p-sele           Selection rate of column on part table              \n\
       --p-bitmap         Flag of  bitmap index on part<1——bitmap 0——vector>  \n\
       --p-groups         Number of groups of columns on part                 \n\
       --c-sele           Selection rate of column on customer table          \n\
       --c-bitmap         Flag of  bitmap index on customer<1——bitmap 0——vector>\n\
       --c-groups         Number of groups of columns on customer             \n\
                                                                              \n\
    Basic user options                                                        \n\
        -h --help         Show this message                                   \n\
    \n");
}
/**
 * @brief command line handling functions
 * 
 * @param argc 
 * @param argv 
 * @param cmd_params 
 * @return void 
 */
void parse_args(int argc, char **argv, param_t *cmd_params)
{

    int c, i, found;
    static int basic_numa_flag;
    while (1)
    {
        static struct option long_options[] =
            {
                /* These options set a flag. */
               

                {"help", no_argument, 0, 'h'},
                /* These options don't set a flag.
                   We distinguish them by their indices. */
                {"nthreads", required_argument, 0, 'n'},
                {"basic_numa", no_argument, &basic_numa_flag, 1},
                {"SF", required_argument, 0, 'F'},
                {"sqlnum", required_argument, 0, 'sq'},
                {"d-bitmap", required_argument, 0, 'd'},
                {"s-bitmap", required_argument, 0, 'f'},
                {"p-bitmap", required_argument, 0, 'P'},
                {"c-bitmap", required_argument, 0, 'c'},
                {"d-sele", required_argument, 0, 'D'},
                {"d-groups", required_argument, 0, 'DG'},
                {"s-sele", required_argument, 0, 'ss'},
                {"s-groups", required_argument, 0, 'sg'},
                {"p-sele", required_argument, 0, 'ps'},
                {"p-groups", required_argument, 0, 'pg'},
                {"c-sele", required_argument, 0, 'cs'},
                {"c-groups", required_argument, 0, 'cg'},
                {0, 0, 0, 0}};
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, "n:hv:F:sq:d:f:P:c:D:DG:ss:sg:ps:pg:cs:cg:dn:sn:pn:cn",
                        long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;
        switch (c)
        {
        case 0:
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
                break;
            printf("option %s", long_options[option_index].name);
            if (optarg)
                printf(" with arg %s", optarg);
            printf("\n");
            break;

        case 'h':
        case '?':
            /* getopt_long already printed an error message. */
            print_help(argv[0]);
            exit(EXIT_SUCCESS);
            break;
        case 'n':
            cmd_params->nthreads = atoi(optarg);
            break;
        case 'F':

            cmd_params->sf = atoi(optarg);

            break;
        case 'sq':

            cmd_params->sqlnum = atoi(optarg);

            break;
        case 'd':

            cmd_params->d_bitmap = atoi(optarg);

            break;
        case 'f':

            cmd_params->s_bitmap = atoi(optarg);

            break;
        case 'P':

            cmd_params->p_bitmap = atoi(optarg);

            break;
        case 'c':

            cmd_params->c_bitmap = atoi(optarg);

            break;
        case 'D':

            cmd_params->d_sele = atof(optarg);

            break;
        case 'DG':

            cmd_params->d_groups = atoi(optarg);

            break;
        case 'ss':

            cmd_params->s_sele = atof(optarg);

            break;
        case 'sg':

            cmd_params->s_groups = atoi(optarg);

            break;
        case 'ps':

            cmd_params->p_sele = atof(optarg);

            break;
        case 'pg':

            cmd_params->p_groups = atoi(optarg);

            break;
        case 'cs':

            cmd_params->c_sele = atof(optarg);

            break;
        case 'cg':

            cmd_params->c_groups = atoi(optarg);

            break;
        default:
            break;
        }
    }
    cmd_params->basic_numa = basic_numa_flag;
    /* Print any remaining command line arguments (not options). */
    if (optind < argc)
    {
        std::cout << "non-option arguments: ";
        while (optind < argc)
            std::cout << argv[optind++] << " ";
        std::cout << std::endl;
    }
}
/**
 * @brief set the join order by select rate
 * 
 * @param sele_array
 * @param orders 
 * @param size 
 * @return void 
 */
void sort_by_rate(double* sele_array, int* orders, int size) {
    for(int i = 0; i < size; i++) {
        for(int j = i + 1; j < size; j++) {
            if(sele_array[i] > sele_array[j]){
                int order_tmp = orders[i];
                orders[i] = orders[j];
                orders[j] = order_tmp;
            }
        }
    }
}



/**
 * @brief Join option based on column-wise model with dynamic vector
 * 
 * @param param 
 * @return void 
 */
void *join_cwm_dv_thread(void *param)
{
    pth_cwmjoint *arg = (pth_cwmjoint *)param;
    
    
    if (!arg->join_id)
    {
        *(arg->index) = 0;
        for (int i = 0; i < arg->num_tuples; i++)
        {
            
            int location = arg->start + i;
            int idx_flag = arg->dimvec[arg->fk[location]];
            if (idx_flag != DIM_NULL)
            {
                arg->OID[*(arg->index)] = location;
                arg->groupID[*(arg->index)] = idx_flag * arg->factor;
                (*(arg->index))++;
                
            }
        }
    }
    else
    {
        
        int comlength = *(arg->index);
        *(arg->index) = 0;
        for (int i = 0; i < comlength; i++)
        {
            int location = arg->OID[i];
            int idx_flag = arg->dimvec[arg->fk[location]];
            if (idx_flag != DIM_NULL)
            {
                arg->OID[*(arg->index)] = location;
                arg->groupID[*(arg->index)] = arg->groupID[i] + idx_flag * arg->factor;
                (*(arg->index))++;
            }
            
            

        }
    }
}
/**
 * @brief Join option based on column-wise model
 * 
 * @param param 
 * @return void 
 */
void *join_cwm_sv_thread(void *param)
{
    pth_cwmjoint *arg = (pth_cwmjoint *)param;
    for (int i = 0; i < arg->num_tuples; i++)
    {
        int location = arg->start + i;
        int idx_flag = arg->dimvec[arg->fk[location]];
        if (!arg->join_id)
        {
            if (idx_flag != DIM_NULL)
            {
                arg->groupID[location] = idx_flag * arg->factor;

            }
            else
                arg->groupID[location] = GROUP_NULL;
        }
        else
        {
            if ((arg->groupID[location] != GROUP_NULL) && (idx_flag != DIM_NULL))
            {
                arg->groupID[location] += idx_flag * arg->factor;
           

            }
            else
                arg->groupID[location] = GROUP_NULL;
        }
        
            
    }
    
}
/**
 * @brief OLAPcore based on row-wise model
 * 
 * @param param 
 * @return void 
 */
void *OLAPcore_row_thread(void *param)
{
    pth_rowolapcoret *arg = (pth_rowolapcoret *)param;
    int groupID = 0;
    for (int i = 0; i < arg->num_tuples; i++)
    {
        int location = i + arg->start;
        
        int flag = 1;
        for (int j = 0; j < arg->dimvec_nums; j++)
        {
            int table_order = arg->orders[j];
            int idx_flag = arg->dimvec_array[table_order][arg->fk_array[table_order][location]];
            
            if (idx_flag != DIM_NULL)
            {
                groupID += idx_flag * arg->factor[j]; 
                continue;
            }
                    
            else
            {
                flag = 0;
                groupID = 0;
                break;
            }

                
        }
        if (flag)
        {
            arg->group_vector[groupID] += arg->M1[i] + arg->M2[i];
            groupID = 0;
        }
       
        
    }

}

/**
 * @brief OLAPcore based on Vector-wise model and dynamic vector
 * 
 * @param param 
 * @return void 
 */
void *OLAPcore_vwm_dv_thread(void *param)
{
    pth_vwmolapcoret *arg = (pth_vwmolapcoret *)param;
    int nblock = arg->num_tuples / size_v;
    int iter = 0;
    while (iter <= nblock)
    {
        int64_t length = (iter == nblock) ? arg->num_tuples % size_v : size_v;
        int64_t comlength;
        for (int i = 0; i < arg->dimvec_nums; i++)
        {
            if (!i)
            {
                *(arg->index) = 0;
                
                for (int j = 0; j < length; j++)
                {
                    int location = arg->start + iter * size_v + j;
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array[table_order][arg->fk_array[table_order][location]];
                    if (idx_flag != DIM_NULL)
                    {
                        arg->OID[*(arg->index)] = location;
                        arg->groupID[*(arg->index)] = idx_flag * arg->factor[i];
                        (*(arg->index)) ++;
                    }
                }

            }
            else
            {
                comlength = *(arg->index);
                *(arg->index) = 0;
                for (int j = 0; j < comlength; j++)
                {
                    int location = arg->OID[j];
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array[table_order][arg->fk_array[table_order][location]];
                    if (idx_flag != DIM_NULL)
                    {
                        arg->OID[*(arg->index)] = location;
                        arg->groupID[*(arg->index)] = arg->groupID[j] + idx_flag * arg->factor[i];
                        (*(arg->index)) ++;
                    }
                }
            }

        }
        comlength = *(arg->index);
        for (int i = 0; i < comlength; i++)
        {
            int16_t tmp = arg->groupID[i];
            int location = arg->OID[i];
            arg->group_vector[tmp] += arg->M1[location] + arg->M2[location];


        }
        iter++;
    }
    

}
/**
 * @brief OLAPcore based on Vector-wise model and dynamic vector for numa test
 * 
 * @param param 
 * @return void 
 */
void *OLAPcore_vwm_dv_numa_thread(void *param)
{
    pth_vwmolapcoret_numa *arg = (pth_vwmolapcoret_numa *)param;
    int nblock = arg->num_tuples / size_v;
    int iter = 0;
    while (iter <= nblock)
    {
        int64_t length = (iter == nblock) ? arg->num_tuples % size_v : size_v;
        int64_t comlength;
        for (int i = 0; i < arg->dimvec_nums; i++)
        {
            if (!i)
            {
                *(arg->index) = 0;
                
                for (int j = 0; j < length; j++)
                {
                    int location = arg->start + iter * size_v + j;
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array_numa->dimvec[table_order][arg->fk_array_numa->fk[table_order][location]];
                    
                    if (idx_flag != DIM_NULL)
                    {
                        arg->OID[*(arg->index)] = location;
                        arg->groupID[*(arg->index)] = idx_flag * arg->factor[i];
                        (*(arg->index)) ++;
                    }
                }

            }
            else
            {
                comlength = *(arg->index);
                *(arg->index) = 0;
                for (int j = 0; j < comlength; j++)
                {
                    int location = arg->OID[j];
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array_numa->dimvec[table_order][arg->fk_array_numa->fk[table_order][location]];
                    if (idx_flag != DIM_NULL)
                    {
                        arg->OID[*(arg->index)] = location;
                        arg->groupID[*(arg->index)] = arg->groupID[j] + idx_flag * arg->factor[i];
                        (*(arg->index)) ++;
                    }
                }
            }

        }
        comlength = *(arg->index);
        for (int i = 0; i < comlength; i++)
        {
            int16_t tmp = arg->groupID[i];
            int location = arg->OID[i];
            arg->group_vector[tmp] += arg->M1[location] + arg->M2[location];

        }
        iter++;
    }
    

}
/**
 * @brief OLAPcore based on Vector-wise model and static vector
 * 
 * @param param 
 * @return void 
 */
void *OLAPcore_vwm_sv_thread(void *param)
{
    pth_vwmolapcoret *arg = (pth_vwmolapcoret *)param;
    int nblock = arg->num_tuples / size_v;
    int iter = 0;
    while (iter <= nblock)
    {
        int64_t length = (iter == nblock) ? arg->num_tuples % size_v : size_v;
        int64_t comlength;
        for (int i = 0; i < arg->dimvec_nums; i++)
        {
            if (!i)
            {
                
                
                for (int j = 0; j < length; j++)
                {
                    int location = arg->start + iter * size_v + j;
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array[table_order][arg->fk_array[table_order][location]];
                    if (idx_flag != DIM_NULL)
                        arg->groupID[j] = idx_flag * arg->factor[i];
                    else
                        arg->groupID[j] = GROUP_NULL;

                }

            }
            else
            {
                for (int j = 0; j < length; j++)
                {
                    int location = arg->start + iter * size_v + j;
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array[table_order][arg->fk_array[table_order][location]];
                    if ((arg->groupID[j] != GROUP_NULL) && (idx_flag != DIM_NULL))
                        arg->groupID[j] += idx_flag * arg->factor[i];
                    else
                        arg->groupID[j] = GROUP_NULL;

                }
            }

        }
        
        for (int i = 0; i < length; i++)
        {
            int16_t tmp = arg->groupID[i];
            int location = arg->start + iter * size_v + i;
            if (tmp != GROUP_NULL)
            {
                arg->group_vector[tmp] += arg->M1[location] + arg->M2[location];
            }

        }
        iter++;
    }
    

}
/**
 * @brief OLAPcore based on Vector-wise model and static vector for numa test
 * 
 * @param param 
 * @return void 
 */
void *OLAPcore_vwm_sv_numa_thread(void *param)
{
    pth_vwmolapcoret_numa *arg = (pth_vwmolapcoret_numa *)param;
    int nblock = arg->num_tuples / size_v;
    int iter = 0;

    while (iter <= nblock)
    {
        int64_t length = (iter == nblock) ? arg->num_tuples % size_v : size_v;
        int64_t comlength;
        for (int i = 0; i < arg->dimvec_nums; i++)
        {
            if (!i)
            {
                
                
                for (int j = 0; j < length; j++)
                {
                    int location = arg->start + iter * size_v + j;
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array_numa->dimvec[table_order][arg->fk_array_numa->fk[table_order][location]];
                    if (idx_flag != DIM_NULL)
                        arg->groupID[j] = idx_flag * arg->factor[i];
                    else
                        arg->groupID[j] = GROUP_NULL;
                    
                }

            }
            else
            {
                for (int j = 0; j < length; j++)
                {
                    int location = arg->start + iter * size_v + j;
                    int table_order = arg->orders[i];
                    int idx_flag = arg->dimvec_array_numa->dimvec[table_order][arg->fk_array_numa->fk[table_order][location]];
                    if ((arg->groupID[j] != GROUP_NULL) && (idx_flag != DIM_NULL))
                        arg->groupID[j] += idx_flag * arg->factor[i];
                    else
                        arg->groupID[j] = GROUP_NULL;
            

                }
            }

        }
        
        for (int i = 0; i < length; i++)
        {
            int16_t tmp = arg->groupID[i];
            int location = arg->start + iter * size_v + i;
            if (tmp != GROUP_NULL)
            {
                arg->group_vector[tmp] += arg->M1[location] + arg->M2[location];
              
              
            }
            

        }
        iter++;
    }

}
/**
 * @brief Aggregation option based on column-wise model and dynamic vector
 * 
 * @param param 
 * @return void 
 */
void *agg_cwm_dv_thread(void *param)
{
    pth_cwmaggt *arg = (pth_cwmaggt *)param;
    for (int i = 0; i < *(arg->index); i++)
    {
        int16_t tmp = arg->groupID[i];
        arg->group_vector[tmp] +=  arg->M1[arg->OID[i]] + arg->M2[arg->OID[i]];
        
    }
}
/**
 * @brief Aggregation option based on column-wise model and static vector
 * 
 * @param param 
 * @return void 
 */
void *agg_cwm_sv_thread(void *param)
{
    pth_cwmaggt *arg = (pth_cwmaggt *)param;
    for (int i = 0; i < arg->num_tuples; i++)
    {
        int location = i + arg->start;
        int16_t tmp = arg->groupID[location];
        if (tmp != GROUP_NULL)
            arg->group_vector[tmp] += arg->M1[location] + arg->M2[location];
            
    
        
    }
}
/**
 * @brief Aggregation option based on Column-wise model and dynamic vector
 * 
 * @param[in] OID
 * @param[in] groupID 
 * @param[in] index 
 * @param[in] size_lineorder 
 * @param[in] M1 
 * @param[in] M2
 * @return int 
 */
int agg_cwm_dv(int64_t ** OID,
                int16_t ** groupID,
                int * index,
                int & size_lineorder,
                int32_t * M1,
                int32_t * M2,
                int group_nums,
                int  nthreads)
{
    int64_t numS, numSthr;
    int rv;
    cpu_set_t set;
    pth_cwmaggt argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    numS = size_lineorder;
    numSthr = numS / nthreads;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    uint32_t *group_vector[nthreads];
    for (int i = 0; i < nthreads; i++)
    {
        group_vector[i] = (uint32_t *)malloc(sizeof(uint32_t) * group_nums);
        memset(group_vector[i], 0, sizeof(uint32_t) * group_nums);
    }
    for (int i = 0; i < nthreads; i++)
    {
        int cpu_idx = i;
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
        argst[i].num_tuples = (i == (nthreads - 1)) ? numS : numSthr;
        argst[i].start = numSthr * i;
        numS -= numSthr;
        argst[i].OID = OID[i];
        argst[i].groupID = groupID[i];
        argst[i].index = &index[i];
        argst[i].M1 = M1;
        argst[i].M2 = M2;
        argst[i].group_vector = group_vector[i];
        rv = pthread_create(&tid[i], &attr, agg_cwm_dv_thread, (void *)&argst[i]);
        if (rv)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }

    }
    for (int j = 0; j < nthreads; j++)
    {
        pthread_join(tid[j], NULL);
    }
    int count = 0;
    for (int i = 1; i < nthreads; i++)
    {
        for (int j = 0; j < group_nums; j++)
        {
            argst[0].group_vector[j] += argst[i].group_vector[j];     
        }
    }
    for (int i = 0; i < group_nums; i++)
    {
        count += argst[0].group_vector[i];
    }
    return count;
}
/**
 * @brief Aggregation option based on Column-wise model and static vector
 * 
 * @param[in] OID
 * @param[in] groupID 
 * @param[in] index 
 * @param[in] size_lineorder 
 * @param[in] M1 
 * @param[in] M2
 * @return int 
 */
int agg_cwm_sv( int16_t * groupID,
                int & size_lineorder,
                int32_t * M1,
                int32_t * M2,
                int group_nums,
                int  nthreads)
{
    int64_t numS, numSthr;
    int rv;
    cpu_set_t set;
    pth_cwmaggt argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    numS = size_lineorder;
    numSthr = numS / nthreads;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    uint32_t *group_vector[nthreads];
    for (int i = 0; i < nthreads; i++)
    {
        group_vector[i] = (uint32_t *)malloc(sizeof(uint32_t) * group_nums);
        memset(group_vector[i], 0, sizeof(uint32_t) * group_nums);
    }
    for (int i = 0; i < nthreads; i++)
    {
        int cpu_idx = i;
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
        argst[i].num_tuples = (i == (nthreads - 1)) ? numS : numSthr;
        argst[i].start = numSthr * i;
        numS -= numSthr;
        argst[i].groupID = groupID;
        argst[i].M1 = M1;
        argst[i].M2 = M2;
        argst[i].group_vector = group_vector[i];
        rv = pthread_create(&tid[i], &attr, agg_cwm_sv_thread, (void *)&argst[i]);
        if (rv)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }

    }
    for (int j = 0; j < nthreads; j++)
    {
        pthread_join(tid[j], NULL);
    }
    int count = 0;
    for (int i = 1; i < nthreads; i++)
    {
        for (int j = 0; j < group_nums; j++)
        {
            argst[0].group_vector[j] += argst[i].group_vector[j];     
        }
    }
    for (int i = 0; i < group_nums; i++)
    {
        count += argst[0].group_vector[i];
    }
    return count;
}
/**
 * @brief join option based on Column-wise model and dynamic vector
 * 
 * @param[in] dimvec_array
 * @param[in] fk_array 
 * @param[in] orders 
 * @param[in] dimvec_nums 
 * @param[out] OID 
 * @param[out] groupID 
 * @param[in] size_lineorder 
 * @return void 
 */
void join_cwm_dv( int8_t ** dimvec_array,
                  int32_t ** fk_array,
                  int * orders,
                  int& dimvec_nums,
                  int64_t ** OID,
                  int16_t ** groupID,
                  int * factor,
                  int * index,
                  int& size_lineorder,
                  int nthreads,
                  int32_t ** M1_array,
                  int32_t ** M2_array)
{
    for (int i = 0; i < dimvec_nums; i++)
    {
        int64_t numS, numSthr;
        int rv;
        cpu_set_t set;
        pth_cwmjoint argst[nthreads];
        pthread_t tid[nthreads];
        pthread_attr_t attr;
        pthread_barrier_t barrier;
        numS = size_lineorder;
        numSthr = numS / nthreads;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, STACKSIZE);
        rv = pthread_barrier_init(&barrier, NULL, nthreads);
        for (int j = 0; j < nthreads; j++)
        {
            int cpu_idx = j;
            CPU_ZERO(&set);
            CPU_SET(cpu_idx, &set);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
            argst[j].num_tuples = (j == (nthreads - 1)) ? numS : numSthr;
            argst[j].start = numSthr * j;
            numS -= numSthr;
            argst[j].dimvec = dimvec_array[orders[i]];
            argst[j].fk = fk_array[orders[i]];
            argst[j].OID = OID[j];
            argst[j].groupID = groupID[j];
            argst[j].factor = factor[i];
            argst[j].tid  = j;
            argst[j].index = & index[j];
            argst[j].join_id = i;
            argst[j].M1 = M1_array[j];
            argst[j].M2 = M2_array[j];
            rv = pthread_create(&tid[j], &attr, join_cwm_dv_thread, (void *)&argst[j]);
            if (rv)
            {
                printf("ERROR; return code from pthread_create() is %d\n", rv);
                exit(-1);
            }
        }
        for (int j = 0; j < nthreads; j++)
        {
            pthread_join(tid[j], NULL);
        }
    }
}


/**
 * @brief join option based on Column-wise model and static vector
 * 
 * @param[in] dimvec_array
 * @param[in] fk_array 
 * @param[in] orders 
 * @param[in] dimvec_nums 
 * @param[out] groupID 
 * @param[in] size_lineorder 
 * @return void 
 */
void join_cwm_sv(int8_t ** dimvec_array,
                  int32_t ** fk_array,
                  int * orders,
                  int& dimvec_nums,
                  int16_t * groupID,
                  int * factor,
                  int& size_lineorder,
                  int nthreads)
{
    for (int i = 0; i < dimvec_nums; i++)
    {
        int64_t numS, numSthr;
        int rv;
        cpu_set_t set;
        pth_cwmjoint argst[nthreads];
        pthread_t tid[nthreads];
        pthread_attr_t attr;
        pthread_barrier_t barrier;
        numS = size_lineorder;
        numSthr = numS / nthreads;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, STACKSIZE);
        rv = pthread_barrier_init(&barrier, NULL, nthreads);
        for (int j = 0; j < nthreads; j++)
        {
            int cpu_idx = j;
            CPU_ZERO(&set);
            CPU_SET(cpu_idx, &set);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
            argst[j].num_tuples = (j == (nthreads - 1)) ? numS : numSthr;
            argst[j].start = numSthr * j;
            numS -= numSthr;
            argst[j].dimvec = dimvec_array[orders[i]];
            argst[j].fk = fk_array[orders[i]];
            argst[j].groupID = groupID;
            argst[j].factor = factor[i];
            argst[j].join_id = i;
            rv = pthread_create(&tid[j], &attr, join_cwm_sv_thread, (void *)&argst[j]);
            if (rv)
            {
                printf("ERROR; return code from pthread_create() is %d\n", rv);
                exit(-1);
            }
        }
        for (int j = 0; j < nthreads; j++)
        {
            pthread_join(tid[j], NULL);
        }
    }
    
}
/**
 * @brief OLAPcore based on Row-wise model
 * 
 * @param[in] dimvec_array
 * @param[in] fk_array 
 * @param[in] orders 
 * @param[in] dimvec_nums 
 * @param[in] factor  
 * @param[in] size_lineorder
 * @param[in] group_nums
 * @param[in] M1
 * @param[in] M2
 * @return void 
 */
int OLAPcore_rowwise(int8_t ** dimvec_array, int32_t ** fk_array,
                  int * orders, int& dimvec_nums, int * factor,
                  int& size_lineorder, int & group_nums,
                  int32_t *M1, int32_t *M2, int  nthreads)
{
    
    int64_t numS, numSthr;
    int  rv;
    cpu_set_t set;
    pth_rowolapcoret argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    numS = size_lineorder;
    numSthr = numS / nthreads;
    uint32_t *group_vector[nthreads];
    for (int i = 0; i < nthreads; i++)
        group_vector[i] = new uint32_t[group_nums];
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    for (int j = 0; j < nthreads; j++)
    {
        int cpu_idx = j;
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
        argst[j].num_tuples = (j == (nthreads - 1)) ? numS : numSthr;
        argst[j].start = numSthr * j;
        numS -= numSthr;
        argst[j].dimvec_array = dimvec_array;
        argst[j].fk_array = fk_array;
        argst[j].factor = factor;
        argst[j].dimvec_nums = dimvec_nums;
        argst[j].group_vector = group_vector[j];
        argst[j].M1 = M1;
        argst[j].M2 = M2;
        argst[j].orders = orders;
        rv = pthread_create(&tid[j], &attr, OLAPcore_row_thread, (void *)&argst[j]);
        if (rv)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
    }
    for (int j = 0; j < nthreads; j++)
    {
        pthread_join(tid[j], NULL);
    }
    int count = 0;
    for (int i = 1; i < nthreads; i++)
    {
        for (int j = 0; j < group_nums; j++)
        {
            argst[0].group_vector[j] += argst[i].group_vector[j];     
        }
    }
    for (int i = 0; i < group_nums; i++)
    {
        count += argst[0].group_vector[i];
        
    }
    return count;
  
}
/**
 * @brief OLAPcore based on Vector-wise model and dynamic vector
 * 
 * @param[in] dimvec_array
 * @param[in] fk_array 
 * @param[in] orders 
 * @param[in] dimvec_nums 
 * @param[in] factor  
 * @param[in] size_lineorder
 * @param[in] group_nums
 * @param[in] M1
 * @param[in] M2
 * @return void 
 */
int OLAPcore_vwm_dv(int8_t ** dimvec_array, int32_t ** fk_array,
                  int * orders, int& dimvec_nums, int * factor,
                  int& size_lineorder, int & group_nums,
                  int32_t *M1, int32_t *M2, int  nthreads)
{
    
    int64_t numS, numSthr;
    int rv;
    cpu_set_t set;
    pth_vwmolapcoret argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    numS = size_lineorder;
    numSthr = numS / nthreads;
    pthread_attr_init(&attr);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    int64_t *OID[nthreads];
    int16_t *groupID[nthreads];
    int *index = new int[nthreads];
    uint32_t *group_vector[nthreads];
    for (int i = 0; i < nthreads; i++)
    {
        OID[i] = new int64_t[size_v];
        groupID[i] = new int16_t[size_v];
        group_vector[i] = new uint32_t[group_nums];
        memset(group_vector[i], 0, sizeof(uint32_t) * group_nums);
    }
    
    for (int j = 0; j < nthreads; j++)
    {
        int cpu_idx = j;
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
        argst[j].num_tuples = (j == (nthreads - 1)) ? numS : numSthr;
        argst[j].start = numSthr * j;
        numS -= numSthr;
        argst[j].dimvec_array = dimvec_array;
        argst[j].fk_array = fk_array;
        argst[j].factor = factor;
        argst[j].dimvec_nums = dimvec_nums;
        argst[j].OID = OID[j];
        argst[j].groupID = groupID[j];
        argst[j].group_vector = group_vector[j];
        argst[j].M1 = M1;
        argst[j].M2 = M2;
        argst[j].index = &index[j];
        argst[j].orders = orders;
        rv = pthread_create(&tid[j], &attr, OLAPcore_vwm_dv_thread, (void *)&argst[j]);
        if (rv)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
    }
    for (int j = 0; j < nthreads; j++)
    {
        pthread_join(tid[j], NULL);
    }
    int count = 0;
    for (int i = 1; i < nthreads; i++)
    {
        for (int j = 0; j < group_nums; j++)
        {
            argst[0].group_vector[j] += argst[i].group_vector[j];     
        }
    }
    for (int i = 0; i < group_nums; i++)
    {
        count += argst[0].group_vector[i];
    }
    return count;
}

/**
 * @brief OLAPcore based on Vector-wise model and dynamic vector for numa test
 * 
 * @param[in] dimvec_array_numa
 * @param[in] fk_array_numa
 * @param[in] orders 
 * @param[in] dimvec_nums 
 * @param[in] factor  
 * @param[in] size_lineorder
 * @param[in] group_nums
 * @param[in] M1_p
 * @param[in] M2_p
 * @return void 
 */
int OLAPcore_vwm_dv_numa(Dimvec_array_numa * dimvec_array_numa, Fk_array_numa * fk_array_numa,
                  int * orders, int& dimvec_nums, int * factor,
                  int& size_lineorder, int & group_nums,
                  int32_t **M1_p, int32_t **M2_p, int  nthreads)
{
    int rv;
    cpu_set_t set;
    pth_vwmolapcoret_numa argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    
    int numa_num = numa_max_node() + 1;
    int64_t numS[numa_num], numSthr[numa_num], nthreadsthr[numa_num];
    for (int i = 0; i < numa_num; i++)
    {
      if (i == 0)
      {

        numS[i] = size_lineorder - (numa_num - 1) * size_lineorder/numa_num;
        nthreadsthr[i] = nthreads - (numa_num - 1) * nthreads/numa_num;
        
      }
      
      else 
      {
        numS[i] = size_lineorder/numa_num;
        nthreadsthr[i] = nthreads / numa_num;

      }
      numSthr[i] = numS[i] / nthreadsthr[i];

    }
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    int64_t *OID[nthreads];
    int16_t *groupID[nthreads];
    int *index = new int[nthreads];
    memset(index, 0, sizeof(int) * nthreads);
    uint32_t *group_vector[nthreads];
    for (int i = 0; i < nthreads; i++)
    {
        int numa_id = get_numa_id(i);
        bind_numa(numa_id);
        OID[i] = (int64_t *)numa_alloc(sizeof(int64_t) * size_v);
        memset(OID[i], 0, size_v * sizeof(int64_t));
        groupID[i] = (int16_t *)numa_alloc(sizeof(int16_t) * size_v);
        memset(groupID[i], 0, size_v * sizeof(int16_t));
        group_vector[i] = (uint32_t *)numa_alloc(sizeof(uint32_t) * group_nums);
        memset(group_vector[i], 0, group_nums * sizeof(uint32_t));
    }
    int nthreads_numa[numa_num] = {0};
    for (int j = 0; j < nthreads; j++)
    {
        int cpu_idx = get_cpuid_bynumaid(j);
        int numa_id = get_numa_id(cpu_idx);
        bind_numa(numa_id);
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
        argst[j].num_tuples = (nthreads_numa[numa_id] == (nthreadsthr[numa_id] - 1)) ? numS[numa_id] : numSthr[numa_id];
        
        argst[j].start = numSthr[numa_id] * nthreads_numa[numa_id];
        numS[numa_id] -= numSthr[numa_id];
        nthreads_numa[numa_id]++;
        argst[j].dimvec_array_numa = &dimvec_array_numa[numa_id];
        argst[j].fk_array_numa = &fk_array_numa[numa_id];
        argst[j].factor = factor;
        argst[j].dimvec_nums = dimvec_nums;
        argst[j].OID = OID[j];
        argst[j].groupID = groupID[j];
        argst[j].group_vector = group_vector[j];
        argst[j].M1 = M1_p[numa_id];
        argst[j].M2 = M2_p[numa_id];
        argst[j].index = &index[j];
        argst[j].orders = orders;
        rv = pthread_create(&tid[j], &attr, OLAPcore_vwm_dv_numa_thread, (void *)&argst[j]);
        if (rv)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
    }
    for (int j = 0; j < nthreads; j++)
    {
        pthread_join(tid[j], NULL);
    }
    int count = 0;
    for (int i = 1; i < nthreads; i++)
    {
        for (int j = 0; j < group_nums; j++)
        {
            argst[0].group_vector[j] += argst[i].group_vector[j];     
        }
    }
    for (int i = 0; i < group_nums; i++)
    {
        count += argst[0].group_vector[i];
    }
    return count;
}
/**
 * @brief OLAPcore based on Vector-wise model and static vector
 * 
 * @param[in] dimvec_array
 * @param[in] fk_array 
 * @param[in] orders 
 * @param[in] dimvec_nums 
 * @param[in] factor  
 * @param[in] size_lineorder
 * @param[in] group_nums
 * @param[in] M1
 * @param[in] M2
 * @return void 
 */
int OLAPcore_vwm_sv(int8_t ** dimvec_array, int32_t ** fk_array,
                  int * orders, int& dimvec_nums, int * factor,
                  int& size_lineorder, int & group_nums,
                  int32_t *M1, int32_t *M2, int  nthreads)
{
    
    int64_t numS, numSthr;
    int i, j, rv;
    cpu_set_t set;
    pth_vwmolapcoret argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    numS = size_lineorder;
    numSthr = numS / nthreads;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    int16_t *groupID[nthreads];
    uint32_t *group_vector[nthreads];
    for (int i = 0; i < nthreads; i++)
    {
        groupID[i] = new int16_t[size_v];
        group_vector[i] = new uint32_t[group_nums];
        memset(group_vector[i], 0, sizeof(uint32_t) * group_nums);
    }
    
    for (int j = 0; j < nthreads; j++)
    {
        int cpu_idx = j;
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
        argst[j].num_tuples = (j == (nthreads - 1)) ? numS : numSthr;
        argst[j].start = numSthr * j;
        numS -= numSthr;
        argst[j].dimvec_array = dimvec_array;
        argst[j].fk_array = fk_array;
        argst[j].factor = factor;
        argst[j].dimvec_nums = dimvec_nums;
        argst[j].groupID = groupID[j];
        argst[j].group_vector = group_vector[j];
        argst[j].M1 = M1;
        argst[j].M2 = M2;
        argst[j].orders = orders;
        rv = pthread_create(&tid[j], &attr, OLAPcore_vwm_sv_thread, (void *)&argst[j]);
        if (rv)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
    }
    for (int j = 0; j < nthreads; j++)
    {
        pthread_join(tid[j], NULL);
    }
    int count = 0;
    for (int i = 1; i < nthreads; i++)
    {
        for (int j = 0; j < group_nums; j++)
        {
            argst[0].group_vector[j] += argst[i].group_vector[j];     
        }
    }
    for (int i = 0; i < group_nums; i++)
    {
        count += argst[0].group_vector[i];
    }
    return count;
}
/**
 * @brief OLAPcore based on Vector-wise model and static vector for numa test
 * 
 * @param[in] dimvec_array_numa
 * @param[in] fk_array_numa
 * @param[in] orders 
 * @param[in] dimvec_nums 
 * @param[in] factor  
 * @param[in] size_lineorder
 * @param[in] group_nums
 * @param[in] M1_p
 * @param[in] M2_p
 * @return void 
 */
int OLAPcore_vwm_sv_numa(Dimvec_array_numa * dimvec_array_numa, Fk_array_numa * fk_array_numa,
                  int * orders, int& dimvec_nums, int * factor,
                  int& size_lineorder, int & group_nums,
                  int32_t **M1_p, int32_t **M2_p, int  nthreads)
{
    
    int  rv;
    cpu_set_t set;
    pth_vwmolapcoret_numa argst[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;

    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, STACKSIZE);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    int16_t *groupID[nthreads];
    uint32_t *group_vector[nthreads];
    int numa_num = numa_max_node() + 1;
    int64_t numS[numa_num], numSthr[numa_num], nthreadsthr[numa_num];
    for (int i = 0; i < numa_num; i++)
    {
      if (i == 0)
      {

        numS[i] = size_lineorder - (numa_num - 1) * size_lineorder/numa_num;
        nthreadsthr[i] = nthreads - (numa_num - 1) * nthreads/numa_num;
        
      }
      
      else 
      {
        numS[i] = size_lineorder/numa_num;
        nthreadsthr[i] = nthreads / numa_num;

      }
      numSthr[i] = numS[i] / nthreadsthr[i];

    }
    for (int i = 0; i < nthreads; i++)
    {
        int numa_id = get_numa_id(i);
        bind_numa(numa_id);
        groupID[i] = (int16_t *)numa_alloc(sizeof(int16_t) * size_v);
        memset(groupID[i], 0, size_v * sizeof(int16_t));
        group_vector[i] = (uint32_t *)numa_alloc(sizeof(uint32_t) * group_nums);
        memset(group_vector[i], 0, group_nums * sizeof(uint32_t));
    }
    
    int nthreads_numa[numa_num] = {0};
    for (int j = 0; j < nthreads; j++)
    {
        int cpu_idx = get_cpuid_bynumaid(j);
        int numa_id = get_numa_id(cpu_idx);
        bind_numa(numa_id);
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);
        argst[j].num_tuples = (nthreads_numa[numa_id] == (nthreadsthr[numa_id] - 1)) ? numS[numa_id] : numSthr[numa_id];
        argst[j].start = numSthr[numa_id] * nthreads_numa[numa_id];
        numS[numa_id] -= numSthr[numa_id];
        nthreads_numa[numa_id]++;
        argst[j].dimvec_array_numa = &dimvec_array_numa[numa_id];
        argst[j].fk_array_numa = &fk_array_numa[numa_id];
        argst[j].factor = factor;
        argst[j].dimvec_nums = dimvec_nums;
        argst[j].groupID = groupID[j];
        argst[j].group_vector = group_vector[j];
        argst[j].M1 = M1_p[numa_id];
        argst[j].M2 = M2_p[numa_id];
        argst[j].orders = orders;
        rv = pthread_create(&tid[j], &attr, OLAPcore_vwm_sv_numa_thread, (void *)&argst[j]);
        if (rv)
        {
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
    }
    for (int j = 0; j < nthreads; j++)
    {
        pthread_join(tid[j], NULL);
    }
    int count = 0;
    for (int i = 1; i < nthreads; i++)
    {
        for (int j = 0; j < group_nums; j++)
        {
            argst[0].group_vector[j] += argst[i].group_vector[j];     
        }
    }
    for (int i = 0; i < group_nums; i++)
    {
        count += argst[0].group_vector[i];
    }
    return count;
}
/**
 * @brief test for OLAPcore based on Row-wise model
 * 
 * @param sele_array 
 * @param dimvec_array
 * @param bitmap_array 
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_rowwise(double SF, double* sele_array, int8_t **dimvec_array, int * bitmap_array, int32_t **fk_array,
                           int32_t* M1, int32_t* M2, int* factor, int* orders, int& dimvec_nums, int& group_nums, int nthreads, int sqlnum, std::ofstream& timefile)
{
    std::cout << ">>> Start test OLAPcore using row-wise model" << std::endl;
    timeval start, end;
    double total_rate = 1;
    for (int i = 0; i < dimvec_nums; i++)
        total_rate *=sele_array[orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    gettimeofday(&start, NULL);
    int count = OLAPcore_rowwise(dimvec_array, fk_array, orders, dimvec_nums, factor, size_lineorder, group_nums, M1, M2, nthreads);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          total time: " << ms << "ms" << std::endl; 
    timefile << sqlnum
             << "\t"
             << "Row-wise query model"
             << "\t"
             << ""
             << "\t"
             << "Row-wise query model"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
}
/**
 * @brief test for OLAPcore based on Column-wise model and dynamic vector
 * 
 * @param sele_array 
 * @param dimvec_array
 * @param bitmap_array 
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_cwm_dv(double SF, double* sele_array, int8_t **dimvec_array, int * bitmap_array, int32_t **fk_array,
                           int32_t* M1, int32_t* M2, int* factor, int* orders, int& dimvec_nums, int& group_nums, int  nthreads, int sqlnum, std::ofstream& timefile)
{
    std::cout << ">>> Start test OLAPcore using Column-wise model and dynamic vector" << std::endl;
    timeval start, end;
    double total_rate = 1;
    for (int i = 0; i < dimvec_nums; i++)
        total_rate *=sele_array[orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int64_t *OID[nthreads];
    int16_t *groupID[nthreads];
    int32_t *M1_array[nthreads];
    int32_t *M2_array[nthreads];
    for (int i = 0; i < nthreads; i++)
    {
        OID[i] = new int64_t[size_lineorder/nthreads + 1];
        groupID[i] = new int16_t[size_lineorder/nthreads + 1];
        M1_array[i] = new int32_t[size_lineorder/nthreads + 1];
        M2_array[i] = new int32_t[size_lineorder/nthreads + 1];
    }
    int *index = new int[nthreads];
    gettimeofday(&start, NULL);
    join_cwm_dv(dimvec_array, fk_array, orders, dimvec_nums, OID, groupID, factor, index, size_lineorder, nthreads, M1_array, M2_array);
    gettimeofday(&end, NULL);
    double ms_join = calc_ms(end, start);

    gettimeofday(&start, NULL);
    int count = agg_cwm_dv(OID, groupID, index, size_lineorder, M1, M2, group_nums, nthreads);
    gettimeofday(&end, NULL);
    double ms_agg = calc_ms(end, start);
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Join time: " << ms_join << "ms" << std::endl;
    std::cout << "          Agg time: " << ms_agg << "ms" << std::endl;  
    std::cout << "          Total time: " << ms_join + ms_agg << "ms" << std::endl; 
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Column-wise query model with dynamic vector"
             << "\t"
             << "join"
             << "\t"
             << ms_join << std::endl;
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Column-wise query model with dynamic vector"
             << "\t"
             << "agg"
             << "\t"
             << ms_agg << std::endl;
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Column-wise query model with dynamic vector"
             << "\t"
             << "total"
             << "\t"
             << ms_join + ms_agg<< std::endl;
}   
/**
 * @brief test for OLAPcore based on Vector-wise model and dynamic vector
 * 
 * @param sele_array 
 * @param dimvec_array
 * @param bitmap_array 
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_vwm_dv(double SF, double* sele_array, int8_t **dimvec_array, int * bitmap_array, int32_t **fk_array,
                           int32_t* M1, int32_t* M2, int* factor, int* orders, int& dimvec_nums, int& group_nums, int  nthreads, int sqlnum, std::ofstream& timefile)
{
    std::cout << ">>> Start test OLAPcore using Vector-wise model and dynamic vector" << std::endl;
    timeval start, end;
    double total_rate = 1;
    for (int i = 0; i < dimvec_nums; i++)
        total_rate *=sele_array[orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    gettimeofday(&start, NULL);
    int count = OLAPcore_vwm_dv(dimvec_array, fk_array, orders, dimvec_nums, factor, size_lineorder, group_nums, M1, M2, nthreads);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Vector-wise query model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Vector-wise query model with dynamic vector"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
  
}   
/**
 * @brief test for OLAPcore based on Vector-wise model and dynamic vector for numa test
 * 
 * @param sele_array 
 * @param dimvec_array_numa
 * @param bitmap_array 
 * @param fk_array_numa
 * @param M1_p
 * @param M2_p
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_vwm_dv_numa(double SF, double* sele_array, Dimvec_array_numa *dimvec_array_numa, int * bitmap_array, Fk_array_numa *fk_array_numa,
                           int32_t** M1_p, int32_t** M2_p, int* factor, int* orders, int& dimvec_nums, int& group_nums, int  nthreads,int sqlnum, std::ofstream& timefile)
{
    std::cout << ">>> Start test OLAPcore using Vector-wise model and dynamic vector for numa test" << std::endl;
    timeval start, end;
    double total_rate = 1;
    for (int i = 0; i < dimvec_nums; i++)
        total_rate *=sele_array[orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    gettimeofday(&start, NULL);
    int count = OLAPcore_vwm_dv_numa(dimvec_array_numa, fk_array_numa, orders, dimvec_nums, factor, size_lineorder, group_nums, M1_p, M2_p, nthreads);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Vector-wise query model"
             << "\t"
             << "dynamic vector"
             << "\t"
             << "Vector-wise query model with dynamic vector and numa-aware data layout"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
  
}  

/**
 * @brief test for OLAPcore based on Column-wise model and static vector
 * 
 * @param sele_array 
 * @param dimvec_array
 * @param bitmap_array 
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_cwm_sv(double SF, double* sele_array, int8_t **dimvec_array, int * bitmap_array, int32_t **fk_array,
                           int32_t* M1, int32_t* M2, int* factor, int* orders, int& dimvec_nums, int& group_nums, int  nthreads, int sqlnum, std::ofstream& timefile)
{
    std::cout << ">>> Start test OLAPcore using Column-wise model and static vector arrow" << std::endl;
    timeval start, end;
    double total_rate = 1;
    for (int i = 0; i < dimvec_nums; i++)
        total_rate *=sele_array[orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    int16_t *groupID = (int16_t *)malloc(sizeof(int16_t) * size_lineorder);
    memset(groupID, 0 ,sizeof(int16_t) * size_lineorder);
    gettimeofday(&start, NULL);
     for (int i = 0; i < dimvec_nums; i++)
    {
        auto column_id = orders[i];
        int8_t* dimvec;
        int32_t* fk;
        switch(column_id){
            case 0:
            dimvec = (int8_t*) read_arrow_column("parquet/customer.parquet", "c_dim");
            fk = (int32_t*) read_arrow_column("parquet/lineorder.parquet", "fk_c");
            break;
            case 1:
            dimvec = (int8_t*) read_arrow_column("parquet/supplier.parquet", "s_dim");
            fk = (int32_t*) read_arrow_column("parquet/lineorder.parquet", "fk_s");
            break;
            case 2:
            dimvec = (int8_t*) read_arrow_column("parquet/part.parquet", "p_dim");
            fk = (int32_t*) read_arrow_column("parquet/lineorder.parquet", "fk_p");
            break;
            case 3:
            dimvec = (int8_t*) read_arrow_column("parquet/date.parquet", "d_dim");
            fk = (int32_t*) read_arrow_column("parquet/lineorder.parquet", "fk_d");
            break;
            default:
            break;
        }
        dimvec_array[column_id] = dimvec;
        fk_array[column_id] = fk;
    }
    join_cwm_sv(dimvec_array, fk_array, orders, dimvec_nums, groupID, factor, size_lineorder, nthreads);
    gettimeofday(&end, NULL);
    double ms_join = calc_ms(end, start);
    gettimeofday(&start, NULL);
    M1 = (int32_t*) read_arrow_column("parquet/lineorder.parquet", "M1");
    M2 = (int32_t*) read_arrow_column("parquet/lineorder.parquet", "M2");
    int count = agg_cwm_sv(groupID, size_lineorder, M1, M2, group_nums, nthreads);
    gettimeofday(&end, NULL);
    double ms_agg = calc_ms(end, start);
    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Join time: " << ms_join << "ms" << std::endl;
    std::cout << "          Agg time: " << ms_agg << "ms" << std::endl; 
    std::cout << "          Total time: " << ms_join + ms_agg << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "static vector"
             << "\t"
             << "Column-wise query model with static vector"
             << "\t"
             << "join"
             << "\t"
             << ms_join << std::endl;
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "static vector"
             << "\t"
             << "Column-wise query model with static vector"
             << "\t"
             << "agg"
             << "\t"
             << ms_agg << std::endl;
    timefile << sqlnum
             << "\t"
             << "Column-wise query model"
             << "\t"
             << "static vector"
             << "\t"
             << "Column-wise query model with static vector"
             << "\t"
             << "total"
             << "\t"
             << ms_join + ms_agg<< std::endl; 
}   
/**
 * @brief test for OLAPcore based on Vector-wise model and static vector
 * 
 * @param sele_array 
 * @param dimvec_array
 * @param bitmap_array 
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_vwm_sv(double SF, double* sele_array, int8_t **dimvec_array, int * bitmap_array, int32_t **fk_array,
                           int32_t* M1, int32_t* M2, int* factor, int* orders, int& dimvec_nums, int& group_nums, int  nthreads, int sqlnum, std::ofstream& timefile)
{
    std::cout << ">>> Start test OLAPcore using Vector-wise model and static vector" << std::endl;
    timeval start, end;
    double total_rate = 1;
    for (int i = 0; i < dimvec_nums; i++)
        total_rate *=sele_array[orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    gettimeofday(&start, NULL);
    int count = OLAPcore_vwm_sv(dimvec_array, fk_array, orders, dimvec_nums, factor, size_lineorder, group_nums, M1, M2, nthreads);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);


    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Vector-wise query model"
             << "\t"
             << "static vector"
             << "\t"
             << "Vector-wise query model with static vector"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;

}   
/**
 * @brief test for OLAPcore based on Vector-wise model and static vector for numa test
 * 
 * @param sele_array 
 * @param dimvec_array_numa
 * @param bitmap_array 
 * @param fk_array_numa
 * @param M1_p
 * @param M2_p
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_vwm_sv_numa(double SF, double* sele_array, Dimvec_array_numa *dimvec_array_numa, int * bitmap_array, Fk_array_numa *fk_array_numa,
                           int32_t** M1_p, int32_t** M2_p, int* factor, int* orders, int& dimvec_nums, int& group_nums, int  nthreads, int sqlnum, std::ofstream& timefile)
{
    std::cout << ">>> Start test OLAPcore using Vector-wise model and static vector for numa test" << std::endl;
    timeval start, end;
    double total_rate = 1;
    for (int i = 0; i < dimvec_nums; i++)
        total_rate *=sele_array[orders[i]];
    int size_lineorder = size_of_table(TABLE_NAME::lineorder, SF);
    gettimeofday(&start, NULL);
    int count = OLAPcore_vwm_sv_numa(dimvec_array_numa, fk_array_numa, orders, dimvec_nums, factor, size_lineorder, group_nums, M1_p, M2_p, nthreads);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);


    std::cout << "          Result count of total selection rate " << total_rate * 100 << "% is " << count << "/" << size_lineorder << std::endl;
    std::cout << "          Total time: " << ms << "ms" << std::endl;
    timefile << sqlnum
             << "\t"
             << "Vector-wise query model"
             << "\t"
             << "static vector"
             << "\t"
             << "Vector-wise query model with static vector and numa-aware data layout"
             << "\t"
             << "total"
             << "\t"
             << ms << std::endl;
}   
/**
 * @brief test for OLAPcore based on Column-wise model
 * 
 * @param sele_array 
 * @param dimvec_array
 * @param bitmap_array 
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_columnwise(double SF,double* sele_array, int8_t **dimvec_array, int * bitmap_array, int32_t **fk_array,
                           int32_t* M1, int32_t* M2, int* factor, int* orders, int& dimvec_nums, int& group_nums, int  nthreads, int sqlnum, std::ofstream& timefile)
{
    /* 1. OLAPcore based on Column-wise model and dynamic vector*/ 
    test_OLAPcore_cwm_dv(SF, sele_array, dimvec_array, bitmap_array, fk_array,
                        M1, M2, factor, orders, dimvec_nums, group_nums, nthreads, sqlnum, timefile);
    /* 2. OLAPcore based on Column-wise model and static vector*/
    test_OLAPcore_cwm_sv(SF, sele_array, dimvec_array, bitmap_array, fk_array,
                        M1, M2, factor, orders, dimvec_nums, group_nums, nthreads, sqlnum, timefile);
}

/**
 * @brief test for OLAPcore based on Vector-wise model
 * 
 * @param sele_array 
 * @param dimvec_array
 * @param bitmap_array 
 * @param fk_array
 * @param M1
 * @param M2
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_vectorwise(double SF, double* sele_array, int8_t **dimvec_array, int * bitmap_array, int32_t **fk_array,
                           int32_t* M1, int32_t* M2, int* factor, int* orders, int& dimvec_nums, int& group_nums, int nthreads, int sqlnum, std::ofstream& timefile)
{
    /* 1. OLAPcore based on Column-wise model and dynamic vector*/ 
    test_OLAPcore_vwm_dv(SF, sele_array, dimvec_array, bitmap_array, fk_array,
                        M1, M2, factor, orders, dimvec_nums, group_nums, nthreads, sqlnum, timefile);
    /* 2. OLAPcore based on Column-wise model and static vector*/
    test_OLAPcore_vwm_sv(SF, sele_array, dimvec_array, bitmap_array, fk_array,
                        M1, M2, factor, orders, dimvec_nums, group_nums, nthreads, sqlnum, timefile);
}
/**
 * @brief test for OLAPcore based on Vector-wise model for numa test
 * 
 * @param sele_array 
 * @param dimvec_array_numa
 * @param bitmap_array 
 * @param fk_array_numa
 * @param M1_p
 * @param M2_p
 * @param factor
 * @param orders
 * @param dimvec_nums
 * @param group_nums
 * @return void 
 */
void test_OLAPcore_vectorwise_numa(double SF, double* sele_array, Dimvec_array_numa *dimvec_array_numa, int * bitmap_array, Fk_array_numa *fk_array_numa,
                           int32_t** M1_p, int32_t** M2_p, int* factor, int* orders, int& dimvec_nums, int& group_nums, int nthreads, int sqlnum, std::ofstream& timefile)
{
    /* 1. OLAPcore based on Column-wise model and dynamic vector for numa test*/ 
    test_OLAPcore_vwm_dv_numa(SF, sele_array, dimvec_array_numa, bitmap_array, fk_array_numa,
                        M1_p, M2_p, factor, orders, dimvec_nums, group_nums, nthreads, sqlnum, timefile);
    /* 2. OLAPcore based on Column-wise model and static vector for numa test*/
    test_OLAPcore_vwm_sv_numa(SF, sele_array, dimvec_array_numa, bitmap_array, fk_array_numa,
                        M1_p, M2_p, factor, orders, dimvec_nums, group_nums, nthreads, sqlnum, timefile);
}
int main(int argc, char **argv)
{

    /* Command line parameters */
    param_t cmd_params;
    double tim_all[3] = {0};
    struct timeval start, end;
    int i, j;
    int index_num=0;
    cmd_params.nthreads = 96;
    cmd_params.sqlnum = 0;
    cmd_params.sf = 1.0;

    cmd_params.d_sele = 0.1;
    cmd_params.s_sele = 0.1;
    cmd_params.p_sele = 0.1;
    cmd_params.c_sele = 0.1;

    cmd_params.d_groups = 10;
    cmd_params.s_groups = 10;
    cmd_params.p_groups = 10;
    cmd_params.c_groups = 10;

    cmd_params.d_bitmap = 1;
    cmd_params.s_bitmap = 1;
    cmd_params.p_bitmap = 1;
    cmd_params.c_bitmap = 1;
    parse_args(argc, argv, &cmd_params);
    
    std::ofstream timefile;
    timefile.open(OLAPCORE_TEST_TIME_FILE,  std::ios::app);
    if (cmd_params.sqlnum == 21)
    {
        timefile << "query number"
             << "\t"
             << "query model"
             << "\t"
             << "intermediate results"
             << "\t"
             << "query model with different intermediate result"
             << "\t"
             << "phase type"
             << "\t"
             << "runtimes(ms)" << std::endl;
    }
    
    int8_t *dimvec_c, *dimvec_s, *dimvec_p, *dimvec_d;
    int32_t *fk_c, *fk_s, *fk_p, *fk_d;
    int32_t *M1, *M2;
    int factor[4]  = {1, 1, 1, 1};
    uint32_t groups[4] = {cmd_params.c_groups, cmd_params.s_groups, cmd_params.p_groups, cmd_params.d_groups};
    gen_data(cmd_params.c_sele, cmd_params.s_sele, cmd_params.p_sele, cmd_params.d_sele, cmd_params.sf, cmd_params.c_bitmap, cmd_params.s_bitmap, cmd_params.p_bitmap, cmd_params.d_bitmap,
            cmd_params.c_groups, cmd_params.s_groups, cmd_params.p_groups, cmd_params.d_groups,
            dimvec_c, dimvec_s, dimvec_p, dimvec_d,
            fk_c, fk_s, fk_p, fk_d,
            M1, M2);
    int dimvec_nums = 0;
    int group_nums = 1;
    double sele_array[4];
    int8_t *dimvec_array[4];
    int bitmap_array[4];
    int32_t *fk_array[4];
    int orders[4];
    /*customer*/
    sele_array[0] = cmd_params.c_sele;
    dimvec_array[0] = dimvec_c;
    bitmap_array[0] = cmd_params.c_bitmap;
    fk_array[0] = fk_c;
    /*supplier*/
    sele_array[1] = cmd_params.s_sele;
    dimvec_array[1] = dimvec_s;
    bitmap_array[1] = cmd_params.s_bitmap;
    fk_array[1] = fk_s;
    /*part*/
    sele_array[2] = cmd_params.p_sele;
    dimvec_array[2] = dimvec_p;
    bitmap_array[2] = cmd_params.p_bitmap;
    fk_array[2] = fk_p;
    /*date*/
    sele_array[3] = cmd_params.d_sele;
    dimvec_array[3] = dimvec_d;
    bitmap_array[3] = cmd_params.d_bitmap;
    fk_array[3] = fk_d;
    for (int i = 0; i < 4; i++)
    {
        if (groups[i] == 0)
            continue;
        else
        {
            
            orders[dimvec_nums] = i;
            dimvec_nums++;
        }
    }
    // set the join order by select rate
    sort_by_rate(sele_array, orders, dimvec_nums);
    for (int i = 0; i < dimvec_nums; i++)
    {
        group_nums *= groups[orders[i]];
        for (int j = i + 1; j < dimvec_nums; j++)
        {
            if (bitmap_array[orders[i]] == 0)
                break;
            factor[i] *= groups[orders[j]];
        }
    
    }
    
    /* 1. OLAPcore based on Row-wise model*/
    test_OLAPcore_rowwise(cmd_params.sf, sele_array, dimvec_array, bitmap_array, fk_array,
                        M1, M2, factor, orders, dimvec_nums, group_nums, cmd_params.nthreads, cmd_params.sqlnum, timefile);
    /* 2. OLAPcore based on Column-wise model*/
    test_OLAPcore_columnwise(cmd_params.sf, sele_array, dimvec_array, bitmap_array, fk_array,
                        M1, M2, factor, orders, dimvec_nums, group_nums, cmd_params.nthreads, cmd_params.sqlnum, timefile);  
    /* 3. OLAPcore based on Vector-wise model*/
    test_OLAPcore_vectorwise(cmd_params.sf, sele_array, dimvec_array, bitmap_array, fk_array,
                        M1, M2, factor, orders, dimvec_nums, group_nums, cmd_params.nthreads, cmd_params.sqlnum, timefile);
    
    // int numa_num = numa_max_node() + 1;
    // int8_t  *dimvec_c_p[numa_num], *dimvec_s_p[numa_num], *dimvec_p_p[numa_num], *dimvec_d_p[numa_num];
    // int32_t * fk_c_p[numa_num], *fk_s_p[numa_num], *fk_p_p[numa_num], *fk_d_p[numa_num];
    // int32_t * M1_p[numa_num], * M2_p[numa_num];
    // /*gen_data(cmd_params.c_sele, cmd_params.s_sele, cmd_params.p_sele, cmd_params.d_sele, cmd_params.sf, cmd_params.c_bitmap, cmd_params.s_bitmap, cmd_params.p_bitmap, cmd_params.d_bitmap,
    //         cmd_params.c_groups, cmd_params.s_groups, cmd_params.p_groups, cmd_params.d_groups,
    //         dimvec_c_p, dimvec_s_p, dimvec_p_p, dimvec_d_p,
    //         fk_c_p, fk_s_p, fk_p_p, fk_d_p,
    //         M1_p, M2_p);*/
    // Dimvec_array_numa *dimvec_array_numa = new Dimvec_array_numa[numa_num];
    // Fk_array_numa * fk_array_numa  = new Fk_array_numa[numa_num];
    // int size_customer = size_of_table(TABLE_NAME::customer, cmd_params.sf);
    // int size_supplier = size_of_table(TABLE_NAME::supplier, cmd_params.sf);
    // int size_part = size_of_table(TABLE_NAME::part, cmd_params.sf);
    // int size_date = size_of_table(TABLE_NAME::date, cmd_params.sf);
    // int size_lineorder = size_of_table(TABLE_NAME::lineorder, cmd_params.sf);
    // int *num_lineorder = new int[numa_num];
    // get_numa_info();
    // for (int i = 0; i < numa_num; i++)
    // {
    //   if (i == numa_num - 1) num_lineorder[i] = size_lineorder - (numa_num - 1) * size_lineorder/numa_num;
    //   else num_lineorder[i] = size_lineorder/numa_num;
    // }
    // for (int i = 0; i < numa_num; i++)
    // {
    //     bind_numa(i);
    //     dimvec_c_p[i] = (int8_t *)numa_alloc(sizeof(int8_t) * size_customer);
    //     dimvec_s_p[i] = (int8_t *)numa_alloc(sizeof(int8_t) * size_supplier);
    //     dimvec_p_p[i] = (int8_t *)numa_alloc(sizeof(int8_t) * size_part);
    //     dimvec_d_p[i] = (int8_t *)numa_alloc(sizeof(int8_t) * size_date);
    //     fk_c_p[i] = (int32_t *)numa_alloc(sizeof(int32_t) * num_lineorder[i]);
    //     fk_s_p[i] = (int32_t *)numa_alloc(sizeof(int32_t) * num_lineorder[i]);
    //     fk_p_p[i] = (int32_t *)numa_alloc(sizeof(int32_t) * num_lineorder[i]);
    //     fk_d_p[i] = (int32_t *)numa_alloc(sizeof(int32_t) * num_lineorder[i]);
    //     M1_p[i] = (int32_t *)numa_alloc(sizeof(int32_t) * num_lineorder[i]);
    //     M2_p[i] = (int32_t *)numa_alloc(sizeof(int32_t) * num_lineorder[i]);
        
    // }
    
    // for (int i = 0; i < numa_num; i++)
    // {
    //     for (int j = 0; j < size_customer; j++)
    //         dimvec_c_p[i][j] = dimvec_c[j];
    //     for (int j = 0; j < size_supplier; j++)
    //         dimvec_s_p[i][j] = dimvec_s[j];
    //     for (int j = 0; j < size_part; j++)
    //         dimvec_p_p[i][j] = dimvec_p[j];
    //     for (int j = 0; j < size_date; j++)
    //         dimvec_d_p[i][j] = dimvec_d[j];
    //     for (int j = 0; j < num_lineorder[i]; j++)
    //     {
    //         fk_c_p[i][j] = fk_c[j + i * num_lineorder[0]];
    //         fk_s_p[i][j] = fk_s[j + i * num_lineorder[0]];
    //         fk_p_p[i][j] = fk_p[j + i * num_lineorder[0]];
    //         fk_d_p[i][j] = fk_d[j + i * num_lineorder[0]];
    //         M1_p[i][j] = 5;
    //         M2_p[i][j] = 5;
    //     }
    // }
    
    delete [] dimvec_c;
    delete [] dimvec_s;
    delete [] dimvec_p;
    delete [] dimvec_d;
    delete [] fk_c;
    delete [] fk_s;
    delete [] fk_p;
    delete [] fk_d;
    delete [] M1;
    delete [] M2;
    /*customer*/
    // sele_array[0] = cmd_params.c_sele;
    // for (int i = 0; i < numa_num; i++)
    // {
    //     dimvec_array_numa[i].dimvec[0] = dimvec_c_p[i];
    //     fk_array_numa[i].fk[0] = fk_c_p[i];
        
    // }
    // bitmap_array[0] = cmd_params.c_bitmap;
    
    // /*supplier*/
    // sele_array[1] = cmd_params.s_sele;
    // for (int i = 0; i < numa_num; i++)
    // {
    //     dimvec_array_numa[i].dimvec[1] = dimvec_s_p[i];
    //     fk_array_numa[i].fk[1] = fk_s_p[i];
    // }
    // bitmap_array[1] = cmd_params.s_bitmap;
    
    // /*part*/
    // sele_array[2] = cmd_params.p_sele;
    // for (int i = 0; i < numa_num; i++)
    // {
    //     dimvec_array_numa[i].dimvec[2] = dimvec_p_p[i];
    //     fk_array_numa[i].fk[2] = fk_p_p[i];
    // }
    // bitmap_array[2] = cmd_params.p_bitmap;
    
    // /*date*/
    // sele_array[3] = cmd_params.d_sele;
    // for (int i = 0; i < numa_num; i++)
    // {
    //     dimvec_array_numa[i].dimvec[3] = dimvec_d_p[i];
    //     fk_array_numa[i].fk[3] = fk_d_p[i];
    // }
    
    // bitmap_array[3] = cmd_params.d_bitmap;
    
    // /*OLAPcore based on Vector-wise model for numa test*/
    // test_OLAPcore_vectorwise_numa(cmd_params.sf, sele_array, dimvec_array_numa, bitmap_array, fk_array_numa,
    //                     M1_p, M2_p, factor, orders, dimvec_nums, group_nums, cmd_params.nthreads, cmd_params.sqlnum, timefile);
    

}


