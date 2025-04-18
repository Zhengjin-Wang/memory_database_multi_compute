#include "../../include/metadata.h"
// #include "../include/time_util.h"
// #include "../include/log_util.h"
#include "../../include/gendata_util.hpp"
#include "../../include/statistical_analysis_util.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <immintrin.h> //immintrin.h
#include <x86intrin.h>
#include <pthread.h>
#include <numa.h>

#define BARRIER_ARRIVE(B,RV)                            \
    RV = pthread_barrier_wait(B);   \
    // printf("RV=%d\n",RV);                       \
    // if(RV !=0 && RV != 1){  \
    //     printf("Couldn't wait on barrier\n");           \
    //     exit(EXIT_FAILURE);                             \
    // }

typedef struct param_t param_t;
struct param_t1
{
    bool is_lsr;
};

/* log-writing */
void log_switch_branch(const Selalgo_Branch &selalgo_branch,std::ofstream &timefile){
    switch ((int)selalgo_branch)
    {
        case 0:
            timefile << "BRANCH_ONE_TWO_THREE"
                     << "\t";
            break;
        case 1:
            timefile << "BRANCH_ONE_TWO"
                     << "\t";
            break;
        case 2:
            timefile << "BRANCH_ONE"
                     << "\t";
            break;
        case 3:
            timefile << "NON_BRANCH"
                      << "\t";
            break;
        default:
            break;
    }
}
void log_write_header(bool is_lsr,
                    std::ofstream &selalgo_model_timefile,
                    std::ofstream &selalgo_model_lsr_timefile,
                    std::ofstream &selalgo_timefile,
                    std::ofstream &casestudy_timefile,
                    std::ofstream &casestudy_lsr_timefile)
{
    if (is_lsr)
    {
        selalgo_model_lsr_timefile << "Branching type"
                                   << "\t"
                                   << "Query processing model with different Intermediate Result Type"
                                   << "\t"
                                   << "Lg(Selection rate)"
                                   << "\t"
                                   << "Runtimes(ms)" << std::endl;
       casestudy_lsr_timefile << "Processing mode"
                              << "\t"
                              << "Query model"
                              << "\t"
                              << "Branching type"
                              << "\t"
                              << "Processing model with different Query model"
                              << "\t"
                              << "Lg(Selection rate)"
                              << "\t"
                              << "Runtimes(ms)" << std::endl;
    }
    else
    {
        selalgo_model_timefile << "Query processing model with different Intermediate Result Type"
                               << "\t"
                               << "Branching type "
                               << "\t"
                               << "Selection rate"
                               << "\t"
                               << "Runtimes(ms)" << std::endl;

        selalgo_timefile << "Query processing model"
                         << "\t"
                         << "Intermediate Result Type"
                         << "\t"
                         << "Branching type"
                         << "\t"
                         << "Query processing model with different Intermediate Result Type and Branching type"
                         << "\t"
                         << "Query processing model with different Branching type"
                         << "\t"
                         << "Selection rate"
                         << "\t"
                         << "Runtimes(ms)" << std::endl;
       casestudy_timefile << "Processing mode"
                          << "\t"
                          << "Query model"
                          << "\t"
                          << "Branching type"
                          << "\t"
                          << "Processing model with different Query model"
                          << "\t"
                          << "Selection rate"
                          << "\t"
                          << "Runtimes(ms)" << std::endl;
    }
}

/**
 * @brief Parsing Command Line Parameters
 * @param argc the number of parameters
 * @param argv the pointer to the string array of parameters
 * @param cmd_params the structure with pre-set parameters
 * @return void
 */
void parse_args(int argc, char **argv, param_t1 *cmd_params)
{
    static bool is_lsr;
}
/**
 * @brief medium selectivity implementations: non-braching code SIMD version
 * @param n the num of tuples
 * @param sd_sv the shared dynamic selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_dsel_shared_val_non_branching_simd(idx n, //result_size
                                           std::vector<T> &sd_sv,
                                           const T *col1,
                                           T *val2)
{
    int curindex_col=0;
    __mmask16 bitmap_sd_sv;
    __m512i value = _mm512_set1_epi32(*val2);
    __mmask16 bitmap_zero_simd = _mm512_int2mask(0);    // __mmask16 _mm512_int2mask (int mask)
    
    if (sd_sv.size() == 0)
    {   
        for (int i = 0; i < n; i=i+16)
        {
            //compare count the number of the selected tuple 
            //and add the index into sd_sv 
            __m512i col1_simd = _mm512_loadu_si512(col1+i);
            bitmap_sd_sv = _mm512_cmp_epi32_mask(col1_simd,value,2);
            if(((int)_ktestz_mask16_u8(bitmap_sd_sv,bitmap_zero_simd))==1){    // unsigned char _ktestz_mask16_u8 (__mmask16 a, __mmask16 b)
                for(int j=0;j<16;j++){
                    if(bitmap_sd_sv & (1<<j)){
                        sd_sv.emplace_back(i+j);
                        curindex_col++;
                    }
                }
            }
        }
    }
    else
    {
        int i=0;
        while (i+15 < n)    // i: sd_sv
        {
            // load data        
            int e0=0;   int e1=0;   int e2=0;   int e3=0;
            int e4=0;   int e5=0;   int e6=0;   int e7=0;
            int e8=0;   int e9=0;   int e10=0;  int e11=0;
            int e12=0;  int e13=0;  int e14=0;  int e15=0;
            
            e0 = col1[sd_sv[i]];    e1 = col1[sd_sv[i+1]];
            e2 = col1[sd_sv[i+2]];  e3 = col1[sd_sv[i+3]];
            e4 = col1[sd_sv[i+4]];  e5 = col1[sd_sv[i+5]];
            e6 = col1[sd_sv[i+6]];  e7 = col1[sd_sv[i+7]];
            e8 = col1[sd_sv[i+8]];  e9 = col1[sd_sv[i+9]];
            e10 = col1[sd_sv[i+10]];    e11 = col1[sd_sv[i+11]];
            e12 = col1[sd_sv[i+12]];    e13 = col1[sd_sv[i+13]];
            e14 = col1[sd_sv[i+14]];    e15 = col1[sd_sv[i+15]];

            __m512i col1_simd = _mm512_set_epi32(e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0);    // __m512i _mm512_set_epi32 
            bitmap_sd_sv = _mm512_cmp_epi32_mask(col1_simd,value,2);

            if((int)(_ktestz_mask16_u8(bitmap_sd_sv,bitmap_zero_simd))){    // unsigned char _ktestz_mask16_u8 (__mmask16 a, __mmask16 b)
                for(int j=0;j<16;j++){
                    if(bitmap_sd_sv & (1<<j)){
                        sd_sv[curindex_col]=sd_sv[i+j];
                        curindex_col++;
                    }
                }
            }

            i=i+16;
        }
        int j=curindex_col;
        while(i<n){
            j += (col1[sd_sv[i]] <= *val2);
            if (curindex_col < j)
            {
                sd_sv[curindex_col] = sd_sv[i];
                curindex_col++;
            }
            i++;
        }   
    }
    // std::cout<<"\nsd_sv = ";
    // for(int k=0;k<curindex_col;k++){
    //     std::cout<<sd_sv[k]<<" ";
    // }
    // std::cout<<std::endl;
    return curindex_col;
}
/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param sd_sv the shared dynamic selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_dsel_shared_val_non_branching(idx n,
                                           std::vector<T> &sd_sv,
                                           const T *col1,
                                           T *val2)
{
    idx i = 0, j = 0;
    idx current_idx = 0;
    if (sd_sv.size() == 0)
    {
        for (i = 0, j = 0; i < n; i++)
        {
            j += (col1[i] <= val2[0]);
            if (j > sd_sv.size())
                sd_sv.emplace_back(i);
        }
    }
    else
    {
        for (i = 0, j = 0; i < n; i++)
        {

            j += (col1[sd_sv[i]] <= *val2);
            if (current_idx < j)
            {
                sd_sv[current_idx] = sd_sv[i];
                current_idx++;
            }
        }
    }
    return j;
}
/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param sf_sv the shared fixed selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_shared_val_non_branching(idx n,
                                           idx *sf_sv,
                                           const T *col1,
                                           T *val2,
                                           idx current_size)
{
    idx i = 0, j = 0;
    idx current_idx = 0;
    if (current_size == 0)
    {
        for (i = 0, j = 0; i < n; i++)
        {
            sf_sv[j] = i;
            j += (col1[i] <= val2[0]);
        }
    }
    else
    {
        for (i = 0, j = 0; i < n; i++)
        {

            j += (col1[sf_sv[i]] <= *val2);
            if (current_idx < j)
            {
                sf_sv[current_idx] = sf_sv[i];
                current_idx++;
            }
        }
    }
    return j;
}
/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param s_bitmap the shared bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
void sel_lt_T_bmp_shared_val_non_branching(idx n,
                                           std::vector<bool> &s_bitmap,
                                           const T *col1,
                                           T *val2,
                                           bool firstflag)
{
    idx i = 0;
    if (firstflag)
    {
        for (i = 0; i < n; i++)
            s_bitmap[i] = (col1[i] <= val2[0]);
    }
    else
    {
        for (i = 0; i < n; i++)
            s_bitmap[i] = (col1[i] <= val2[0]) && s_bitmap[i];
    }
    // std::cout<<"bitmap = ";
    // for(int j=0;j<n;j++){
    //     std::cout<<s_bitmap[j]<<"\t";
    // }
    // std::cout<<std::endl;
    return;
}
/**
 * @brief medium selectivity implementations: non-braching code SIMD version
 * @param n the num of tuples
 * @param s_bitmap the shared bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return set s_bitmap_simd
 */
void sel_lt_T_bmp_shared_val_non_branching_simd(idx n, //(16, s_bitmap_simd, Rc, conditions[select_idx], false);
                                           __mmask16 &s_bitmap_simd,
                                           const T *col1,
                                           T *val2,
                                           bool firstflag)
{
    // int bitmap_simd0_value = 0;    // for test

    __m512i value = _mm512_set1_epi32(*val2); //val: condition array
    __mmask16 bitmap_simd1;

    if(firstflag){
        __m512i col_simd0 = _mm512_loadu_si512(col1);
        s_bitmap_simd = _mm512_cmp_epi32_mask(col_simd0,value,2);
        // bitmap_simd0_value = _mm512_mask2int(s_bitmap_simd);    // int _mm512_mask2int (__mmask16 k1)
    }
    else{
        __m512i col_simd1 = _mm512_loadu_si512(col1);
        bitmap_simd1 = _mm512_cmp_epi32_mask(col_simd1,value,2);
        s_bitmap_simd = _kand_mask16(s_bitmap_simd,bitmap_simd1);
        // bitmap_simd0_value = _mm512_mask2int(s_bitmap_simd);    // int _mm512_mask2int (__mmask16 k1) 
    }
}
/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param id_sv the independent dynamic selection vector
 * @return void
 */
void sel_lt_T_dsel_independent_val_non_branching(idx n,
                                                 std::vector<T> &res,
                                                 const T *col1,
                                                 T *val2,
                                                 std::vector<T> &id_sv)
{
    idx i = 0, j = 0;
    if (id_sv.size() == 0)
    {
        for (i = 0, j = 0; i < n; i++)
        {
            j += (col1[i] <= val2[0]);
            if (j > res.size())
                res.emplace_back(i);
        }
    }
    else
    {
        for (i = 0, j = 0; i < n; i++)
        {

            j += (col1[id_sv[i]] <= *val2);
            if (j > res.size())
                res.emplace_back(id_sv[i]);
        }
    }
    return;
}
/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param id_sv the independent dynamic selection vector
 * @return void
 */
void sel_lt_T_dsel_independent_val_non_branching_simd(idx n,
                                                 std::vector<T> &res,
                                                 const T *col1,
                                                 T *val2,
                                                 std::vector<T> &id_sv)
{
    __m512i value = _mm512_set1_epi32 (*val2);
    __m512i col1_simd;
    __mmask16 bitmap_compare_independent,bitmap_zero_simd;
    bitmap_zero_simd = _mm512_int2mask(0);
    
    if(id_sv.size()==0){
        for(int i=0;i<n/16;i++){
            col1_simd = _mm512_loadu_si512(col1+i*16);
            bitmap_compare_independent = _mm512_cmp_epi32_mask(col1_simd,value,2);    //compare
            //add index into res
            if((int)(_ktestz_mask16_u8(bitmap_compare_independent,bitmap_zero_simd))){
                for(int j=0;j<16;j++){
                    if(bitmap_compare_independent & (1<<j)){
                        res.emplace_back(i*16+j);
                    }
                }
            }
        }
    }
    else{
        int i=0;
        while(i<n/16){
            // load data        
            int e0=0;   int e1=0;   int e2=0;   int e3=0;
            int e4=0;   int e5=0;   int e6=0;   int e7=0;
            int e8=0;   int e9=0;   int e10=0;  int e11=0;
            int e12=0;  int e13=0;  int e14=0;  int e15=0;
            
            e0 = col1[id_sv[i*16]];    e1 = col1[id_sv[i*16+1]];
            e2 = col1[id_sv[i*16+2]];  e3 = col1[id_sv[i*16+3]];
            e4 = col1[id_sv[i*16+4]];  e5 = col1[id_sv[i*16+5]];
            e6 = col1[id_sv[i*16+6]];  e7 = col1[id_sv[i*16+7]];
            e8 = col1[id_sv[i*16+8]];  e9 = col1[id_sv[i*16+9]];
            e10 = col1[id_sv[i*16+10]];    e11 = col1[id_sv[i*16+11]];
            e12 = col1[id_sv[i*16+12]];    e13 = col1[id_sv[i*16+13]];
            e14 = col1[id_sv[i*16+14]];    e15 = col1[id_sv[i*16+15]];

            col1_simd = _mm512_set_epi32(e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0);    // __m512i _mm512_set_epi32 
            bitmap_compare_independent = _mm512_cmp_epi32_mask(col1_simd,value,2);

            if((int)(_ktestz_mask16_u8(bitmap_compare_independent,bitmap_zero_simd))){
                for(int j=0;j<16;j++){
                    if(bitmap_compare_independent & (1<<j)){
                        res.emplace_back(id_sv[i*16+j]);
                    }
                }
            }

            i++;
        }
        i=i*16;
        int j=res.size();
        while(i<n){
            j += (col1[id_sv[i]] <= *val2);
            if (j > res.size())
            {
                res.emplace_back(id_sv[i]);
            }
            i++;
        }
    }

    return;
}

/**
 * @brief divide into slice(1024) 
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param id_sv the independent dynamic selection vector
 * @return void
 */
int sel_lt_T_dsel_independent_val_non_branching_simd_gather(bool flag_col,
                                                idx n,  // data length
                                                 int v_n,    // bitmap length
                                                 int *res,
                                                 const T *col1,
                                                 T *val2,
                                                 const int *id_sv)
{
    int current_idx=0;
    __m512i value = _mm512_set1_epi32 (*val2);
    __m512i col1_simd;
    __mmask16 bitmap_compare_independent,bitmap_zero_simd;
    bitmap_zero_simd = _mm512_int2mask(0);

    // printf("v_n=%d\n",v_n);
    
    if(v_n==0){
        if(flag_col){
            for(int i=0;i<n/16;i++){
                col1_simd = _mm512_loadu_si512(col1+i*16);
                bitmap_compare_independent = _mm512_cmp_epi32_mask(col1_simd,value,2);    // compare
                // int bitmap_compare_independent_value = _mm512_mask2int(bitmap_compare_independent);
                // printf("bitmap_value=%d\n",bitmap_compare_independent);
                if((int)(_ktestz_mask16_u8(bitmap_compare_independent,bitmap_zero_simd))){    // add index into res
                    for(int j=0;j<16;j++){
                        if(bitmap_compare_independent & (1<<j)){
                            res[current_idx]=i*16+j;
                            current_idx++;
                            // res.emplace_back(i*16+j);
                        }
                    }
                }
            }
        }
        else{
            return 0;
        }
    }
    else{
        int i=0;
        for(;i<v_n/16;i++){
            __m512i offset_index = _mm512_loadu_si512(id_sv+i*16);
            col1_simd = _mm512_i32gather_epi32(offset_index,col1,4);
            
            bitmap_compare_independent = _mm512_cmp_epi32_mask(col1_simd,value,2);    // compare
            
            if((int)(_ktestz_mask16_u8(bitmap_compare_independent,bitmap_zero_simd))){    // add index into res
                for(int j=0;j<16;j++){
                    if(bitmap_compare_independent & (1<<j)){
                        res[current_idx]=id_sv[i*16+j];
                        current_idx++;
                        // res.emplace_back(i*16+j);
                    }
                }
            }
        }
        i=i*16;
        if(i<v_n){
            while(i<v_n){
                if(col1[id_sv[i]] <= *val2){
                    res[current_idx]=id_sv[i];
                    current_idx++;
                }
                i++;
            }
        }
    }
    return current_idx;
}

/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param if_sv the independent fixed selection vector
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_independent_val_non_branching(idx n,
                                                int *res,
                                                const T *col1,
                                                T *val2,
                                                int *if_sv)
{
    idx i = 0, j = 0;
    if (if_sv == NULL)
    {
        for (i = 0, j = 0; i < n; i++)
        {
            res[j] = i;
            j += (col1[i] <= val2[0]);
        }
    }
    else
    {
        for (i = 0, j = 0; i < n; i++)
        {
            res[j] = if_sv[i];
            j += (col1[if_sv[i]] <= *val2);
        }
    }
    return j;
}

/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param i_bitmap the independent bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return void
 */
void sel_lt_T_bmp_independent_val_non_branching(idx n,
                                                std::vector<bool> &i_bitmap,
                                                const T *col1,
                                                T *val2)
{
    idx i = 0;

    for (i = 0; i < n; i++)
    {
        i_bitmap[i] = (col1[i] <= val2[0]);
    }

    return;
}
/**
 * @brief medium selectivity implementations: non-braching code
 * @param n the num of tuples
 * @param i_bitmap the independent bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return void
 */
void sel_lt_T_bmp_independent_val_non_branching_simd(idx n,
                                                std::vector<__mmask16> &i_bitmap,
                                                const T *col1,
                                                T *val2)
{
    int R_v;
    __m512i value = _mm512_set1_epi32(*val2);
    for (int i = 0; i < n; i++)
    {
        __m512i col_simd = _mm512_loadu_si512(col1+i*16);
        i_bitmap[i] = _mm512_cmp_epi32_mask(col_simd,value,2);
    }

    return;
}
/**
 * @brief medium selectivity implementations: braching code
 * @param n the num of tuples
 * @param sd_sv the shared dynamic selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_dsel_shared_val_branching(idx n,
                                       std::vector<T> &sd_sv,
                                       const T *col1,
                                       T *val2)
{
    idx i = 0, j = 0;
    if (sd_sv.size() == 0)
    {
        for (i = 0, j = 0; i < n; i++)
        {
            if (col1[i] <= *val2)
            {
                j++;
                sd_sv.emplace_back(i);
            }
        }
    }
    else
    {
        for (i = 0, j = 0; i < n; i++)
        {
            if (col1[sd_sv[i]] <= *val2)
            {
                sd_sv[j++] = sd_sv[i];
            }
        }
    }
    return j;
}
/**
 * @brief medium selectivity implementations: braching code
 * @param n the num of tuples
 * @param s_bitmap the shared bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return void
 */
void sel_lt_T_bmp_shared_val_branching(idx n,
                                       std::vector<bool> &s_bitmap,
                                       const T *col1,
                                       T *val2,
                                       bool firstflag)
{
    idx i = 0;
    if (firstflag)
    {
        for (i = 0; i < n; i++)
        {
            if (col1[i] <= *val2)
            {
                s_bitmap[i] = 1;
            }
            else
            {
                s_bitmap[i] = 0;
            }
        }
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            if (s_bitmap[i] && (col1[i] <= *val2))
            {
                s_bitmap[i] = 1;
            }
            else
            {
                s_bitmap[i] = 0;
            }
        }
    }
}

/**
 * @brief medium selectivity implementations: braching code
 * @param n the num of tuples
 * @param sf_sv the shared fixed selection vector
 * @param col1 the filter column
 * @param val2 the filter value
 * @param current_size the length of the selection vector
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_shared_val_branching(idx n,
                                       idx *sf_sv,
                                       const T *col1,
                                       T *val2,
                                       idx current_size)
{
    idx i = 0, j = 0;
    if (current_size == 0)
    {
        for (i = 0, j = 0; i < n; i++)
        {
            if (col1[i] <= *val2)
            {
                sf_sv[j++] = i;
            }
        }
    }
    else
    {
        for (i = 0, j = 0; i < n; i++)
        {
            if (col1[sf_sv[i]] <= *val2)
            {
                sf_sv[j++] = sf_sv[i];
            }
        }
    }
    return j;
}

/**
 * @brief medium selectivity implementations:braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param id_sv the independent dynamic selection vector
 * @return void
 */
void sel_lt_T_dsel_independent_val_branching(idx n,
                                             std::vector<T> &res,
                                             const T *col1,
                                             T *val2,
                                             std::vector<T> &id_sv)
{
    idx i = 0;
    if (id_sv.size() == 0)
    {
        for (i = 0; i < n; i++)
        {
            if (col1[i] <= *val2)
            {
                res.emplace_back(i);
            }
        }
    }
    else
    {
        for (i = 0; i < n; i++)
        {
            if (col1[id_sv[i]] <= *val2)
            {
                res.emplace_back(id_sv[i]);
            }
        }
    }
    return;
}

/**
 * @brief medium selectivity implementations:braching code
 * @param n the num of tuples
 * @param res the selection vector materialization results
 * @param col1 the filter column
 * @param val2 the filter value
 * @param if_sv the independent fixed selection vector
 * @return int the count of dynamic selection vector
 */
idx sel_lt_T_fsel_independent_val_branching(idx n,
                                            int *res,
                                            const T *col1,
                                            T *val2,
                                            int *if_sv)
{
    idx i = 0, j = 0;
    if (if_sv == NULL)
    {
        for (i = 0, j = 0; i < n; i++)
        {
            if (col1[i] <= *val2)
                res[j++] = i;
        }
    }
    else
    {
        for (i = 0, j = 0; i < n; i++)
        {
            if (col1[if_sv[i]] <= *val2)
                res[j++] = if_sv[i];
        }
    }
    return j;
}
/**
 * @brief medium selectivity implementations:braching code
 * @param n the num of tuples
 * @param i_bitmap the independent bitmap
 * @param col1 the filter column
 * @param val2 the filter value
 * @return void
 */
void sel_lt_T_bmp_independent_val_branching(idx n,
                                            std::vector<bool> &i_bitmap,
                                            const T *col1,
                                            T *val2)
{
    idx i = 0;

    for (i = 0; i < n; i++)
    {
        if (col1[i] <= *val2)
            i_bitmap[i] = 1;
        else
            i_bitmap[i] = 0;
    }

    return;
}
/**
 * @brief perform select using Row-wise query processing model
 * @param condition determine the selection rate
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_rowwise(idx condition, const idx &size_R,
                    const T *Ra,
                    const T *Rb,
                    const T *Rc,
                    const T *Rd)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    for (i = 0; i != result_size; ++i)
    {
        if (Ra[i] <= condition && Rb[i] <= condition && Rc[i] <= condition)
        {
            count += Rd[i];
        }
    }
    return count;
}
/**
 * @brief perform select using Culomn-wise query processing model with shared dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param sd_sv the shared dynamic selection vector
 * @return int the count of selection result
 */
idx selalgo_cwm_dsv_shared(idx condition, const idx &size_R,
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd,
                           std::vector<T> &sd_sv,
                           const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    idx current_size_ra = 0;
    idx current_size_rb = 0;
    idx current_size_rc = 0;
    if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
    {
        current_size_ra = sel_lt_T_dsel_shared_val_non_branching(result_size, sd_sv, Ra, &condition);
        current_size_rb = sel_lt_T_dsel_shared_val_non_branching(current_size_ra, sd_sv, Rb, &condition);
        current_size_rc = sel_lt_T_dsel_shared_val_non_branching(current_size_rb, sd_sv, Rc, &condition);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sd_sv[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        current_size_ra = sel_lt_T_dsel_shared_val_branching(result_size, sd_sv, Ra, &condition);
        current_size_rb = sel_lt_T_dsel_shared_val_branching(current_size_ra, sd_sv, Rb, &condition);
        current_size_rc = sel_lt_T_dsel_shared_val_branching(current_size_rb, sd_sv, Rc, &condition);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sd_sv[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        current_size_ra = sel_lt_T_dsel_shared_val_branching(result_size, sd_sv, Ra, &condition);
        current_size_rb = sel_lt_T_dsel_shared_val_branching(current_size_ra, sd_sv, Rb, &condition);
        current_size_rc = sel_lt_T_dsel_shared_val_non_branching(current_size_rb, sd_sv, Rc, &condition);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sd_sv[i]];
        }
    }
    else
    {
        current_size_ra = sel_lt_T_dsel_shared_val_branching(result_size, sd_sv, Ra, &condition);
        current_size_rb = sel_lt_T_dsel_shared_val_non_branching(current_size_ra, sd_sv, Rb, &condition);
        current_size_rc = sel_lt_T_dsel_shared_val_non_branching(current_size_rb, sd_sv, Rc, &condition);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sd_sv[i]];
        }
    }
    return count;
}
/**
 * @brief perform select using Culomn-wise query processing model with shared dynamic selection vector SIMD version
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param sd_sv the shared dynamic selection vector
 * @return int the count of selection result
 */
idx selalgo_cwm_dsv_shared_simd(idx condition, const idx &size_R,
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd,
                           std::vector<T> &sd_sv
                           )
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    idx current_size_ra = 0;
    idx current_size_rb = 0;
    idx current_size_rc = 0;

    current_size_ra = sel_lt_T_dsel_shared_val_non_branching_simd(result_size, sd_sv, Ra, &condition);
    current_size_rb = sel_lt_T_dsel_shared_val_non_branching_simd(current_size_ra, sd_sv, Rb, &condition);
    current_size_rc = sel_lt_T_dsel_shared_val_non_branching_simd(current_size_rb, sd_sv, Rc, &condition);
    // std::cout<<"\nsd_sv = ";
    // int curindex_col=current_size_rc;
    // for(int k=0;k<curindex_col;k++){
    //     std::cout<<sd_sv[k]<<" ";
    // }
    // std::cout<<std::endl;

    // printf("current_size_ra = %d\n",current_size_ra);
    // printf("current_size_rb = %d\n",current_size_rb);
    // printf("current_size_rc = %d\n",current_size_rc);
    return current_size_rc;
    // return current_size_rc;
}
/**
 * @brief perform select using Culomn-wise query processing model with shared fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param sf_sv the shared fixed selection vector
 * @return int the count of selection result
 */
idx selalgo_cwm_fsv_shared(idx condition, const idx &size_R,
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd,
                           idx *sf_sv,
                           const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    idx current_size_ra = 0;
    idx current_size_rb = 0;
    idx current_size_rc = 0;
    if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
    {
        current_size_ra = sel_lt_T_fsel_shared_val_non_branching(result_size, sf_sv, Ra, &condition, 0);
        current_size_rb = sel_lt_T_fsel_shared_val_non_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
        current_size_rc = sel_lt_T_fsel_shared_val_non_branching(current_size_rb, sf_sv, Rc, &condition, current_size_rb);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        current_size_ra = sel_lt_T_fsel_shared_val_branching(result_size, sf_sv, Ra, &condition, 0);
        current_size_rb = sel_lt_T_fsel_shared_val_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
        current_size_rc = sel_lt_T_fsel_shared_val_branching(current_size_rb, sf_sv, Rc, &condition, current_size_ra);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        current_size_ra = sel_lt_T_fsel_shared_val_branching(result_size, sf_sv, Ra, &condition, 0);
        current_size_rb = sel_lt_T_fsel_shared_val_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
        current_size_rc = sel_lt_T_fsel_shared_val_non_branching(current_size_rb, sf_sv, Rc, &condition, current_size_ra);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    else
    {
        current_size_ra = sel_lt_T_fsel_shared_val_branching(result_size, sf_sv, Ra, &condition, 0);
        current_size_rb = sel_lt_T_fsel_shared_val_non_branching(current_size_ra, sf_sv, Rb, &condition, current_size_ra);
        current_size_rc = sel_lt_T_fsel_shared_val_non_branching(current_size_rb, sf_sv, Rc, &condition, current_size_ra);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    return count;
}
/**
 * @brief perform select using Culomn-wise query processing model with shared bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param s_bitmap the shared bitmap
 * @return int the count of selection result
 */
idx selalgo_cwm_bmp_shared(idx condition, const idx &size_R,
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd,
                           std::vector<bool> &s_bitmap,
                           const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
    {
        sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Ra, &condition, true);
        sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rb, &condition, false);
        sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
        for (i = 0; i < result_size; i++)
        {
            if (s_bitmap[i])
                count += Rd[i];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Ra, &condition, true);
        sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Rb, &condition, false);
        sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Rc, &condition, false);
        for (i = 0; i < result_size; i++)
        {
            if (s_bitmap[i])
                count += Rd[i];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Ra, &condition, true);
        sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Rb, &condition, false);
        sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
        for (i = 0; i < result_size; i++)
        {
            if (s_bitmap[i])
                count += Rd[i];
        }
    }
    else
    {
        sel_lt_T_bmp_shared_val_branching(result_size, s_bitmap, Ra, &condition, true);
        sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rb, &condition, false);
        sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
        for (i = 0; i < result_size; i++)
        {
            if (s_bitmap[i])
                count += Rd[i];
        }
    }
    return count;
}

/**
 * @brief perform select using Culomn-wise & Vector-wise query processing model with shared bitmap SIMD version
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param s_bitmap the shared bitmap
 * @return int the count of selection result
 */
idx selalgo_cwm_bmp_shared_simd(idx condition, const idx &size_R, // size_R=16
                           const T *Ra,
                           const T *Rb,
                           const T *Rc,
                           const T *Rd)
{
    __mmask16 s_bitmap;
    idx result_size = 16;    // 16,一次一组
    int count_shared=0;

    // for SIMD, there is no different between branch and non_branch
    sel_lt_T_bmp_shared_val_non_branching_simd(result_size, s_bitmap, Ra, &condition, true);
    sel_lt_T_bmp_shared_val_non_branching_simd(result_size, s_bitmap, Rb, &condition, false);
    sel_lt_T_bmp_shared_val_non_branching_simd(result_size, s_bitmap, Rc, &condition, false);

    for(int k=0;k<16;k++){
        if(s_bitmap & (1<<k)){
            count_shared+=1;
        }
    }

    return count_shared;
}

/**
 * @brief perform select using Culomn-wise query processing model with independent dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_dsv_independent(idx condition, const idx &size_R,
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                std::vector<T> &d_sv_ra,
                                std::vector<T> &d_sv_rb,
                                std::vector<T> &d_sv_rc,
                                const Selalgo_Branch &selalgo_branch)
{
    idx count = 0, count1 = 0;
    idx i;
    idx result_size = size_R;
    if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
    {
        sel_lt_T_dsel_independent_val_non_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_non_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
        for (i = 0; i < d_sv_rc.size(); i++)
        {
            count += Rd[d_sv_rc[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {

        sel_lt_T_dsel_independent_val_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
        for (i = 0; i < d_sv_rc.size(); i++)
        {
            count += Rd[d_sv_rc[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        sel_lt_T_dsel_independent_val_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
        for (i = 0; i < d_sv_rc.size(); i++)
        {
            count += Rd[d_sv_rc[i]];
        }
    }
    else
    {
        sel_lt_T_dsel_independent_val_branching(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_non_branching(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
        sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
        for (i = 0; i < d_sv_rc.size(); i++)
        {
            count += Rd[d_sv_rc[i]];
        }
    }
    return count;
}
/**
 * @brief perform select using Culomn-wise query processing model with independent dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_dsv_independent_simd(idx condition, const idx &size_R,
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                std::vector<T> &d_sv_ra,
                                std::vector<T> &d_sv_rb,
                                std::vector<T> &d_sv_rc)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;

    sel_lt_T_dsel_independent_val_non_branching_simd(result_size, d_sv_ra, Ra, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_non_branching_simd(d_sv_ra.size(), d_sv_rb, Rb, &condition, d_sv_ra);
    sel_lt_T_dsel_independent_val_non_branching_simd(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
    // for (i = 0; i < d_sv_rc.size(); i++)
    // {
    //     count += Rd[d_sv_rc[i]];
    // }
    count = d_sv_rc.size();
    return count;
}

/**
 * @brief perform select using Culomn-wise query processing model with independent dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_dsv_independent_simd_gather(idx condition, const idx &size_R,
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd)
{
   idx count = 0;
    T* d_sv_ra = new T[1024];
    T* d_sv_rb = new T[1024];
    T* d_sv_rc = new T[1024];
    
    int d_sv_ra_count=0;    int d_sv_rb_count=0;    int d_sv_rc_count=0;
    
    d_sv_ra_count = sel_lt_T_dsel_independent_val_non_branching_simd_gather(true,1024, 0, d_sv_ra, Ra, &condition, d_sv_ra);
    d_sv_rb_count = sel_lt_T_dsel_independent_val_non_branching_simd_gather(false,1024, d_sv_ra_count, d_sv_rb, Rb, &condition, d_sv_ra);
    d_sv_rc_count = sel_lt_T_dsel_independent_val_non_branching_simd_gather(false,1024, d_sv_rb_count, d_sv_rc, Rc, &condition, d_sv_rb);

    delete[] d_sv_ra;
    delete[] d_sv_rb;
    delete[] d_sv_rc;
    return d_sv_rc_count;
}

/**
 * @brief perform select using Culomn-wise query processing model with independent fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_fsv_independent(idx condition, const idx &size_R,
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                int *f_sv_ra,
                                int *f_sv_rb,
                                int *f_sv_rc,
                                const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    idx current_size_ra = 0;
    idx current_size_rb = 0;
    idx current_size_rc = 0;
    if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
    {
        current_size_ra = sel_lt_T_fsel_independent_val_non_branching(result_size, f_sv_ra, Ra, &condition, NULL);
        current_size_rb = sel_lt_T_fsel_independent_val_non_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
        current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[f_sv_rc[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        current_size_ra = sel_lt_T_fsel_independent_val_branching(result_size, f_sv_ra, Ra, &condition, NULL);
        current_size_rb = sel_lt_T_fsel_independent_val_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
        current_size_rc = sel_lt_T_fsel_independent_val_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[f_sv_rc[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        current_size_ra = sel_lt_T_fsel_independent_val_branching(result_size, f_sv_ra, Ra, &condition, NULL);
        current_size_rb = sel_lt_T_fsel_independent_val_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
        current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[f_sv_rc[i]];
        }
    }
    else
    {
        current_size_ra = sel_lt_T_fsel_independent_val_branching(result_size, f_sv_ra, Ra, &condition, NULL);
        current_size_rb = sel_lt_T_fsel_independent_val_non_branching(current_size_ra, f_sv_rb, Rb, &condition, f_sv_ra);
        current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[f_sv_rc[i]];
        }
    }
    return count;
}
/**
 * @brief perform select using Culomn-wise query processing model with independent bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_bmp_independent(idx condition, const idx &size_R, //size_R : the number of one group
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                std::vector<bool> &bitmap_Ra,
                                std::vector<bool> &bitmap_Rb,
                                std::vector<bool> &bitmap_Rc,
                                std::vector<bool> &bitmap,
                                const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    if (selalgo_branch == Selalgo_Branch::NON_BRANCH)
    {
        sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Ra, Ra, &condition);
        sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rb, Rb, &condition);
        sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rc, Rc, &condition);
        for (i = 0; i != result_size; ++i)
        {
             if (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i])
                 bitmap[i] = 1;
            else
                 bitmap[i] = 0;
        }
        for (i = 0; i < result_size; i++)
        {
            if (bitmap[i])
                count += Rd[i];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Ra, Ra, &condition);
        sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rb, Rb, &condition);
        sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rc, Rc, &condition);
        for (i = 0; i != result_size; ++i)
        {
            if (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i])
                bitmap[i] = 1;
            else
                bitmap[i] = 0;
        }
        for (i = 0; i < result_size; i++)
        {
            if (bitmap[i])
                count += Rd[i];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Ra, Ra, &condition);
        sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rb, Rb, &condition);
        sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rc, Rc, &condition);
        for (i = 0; i != result_size; ++i)
        {
            if (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i])
                bitmap[i] = 1;
            else
                bitmap[i] = 0;
        }
        for (i = 0; i < result_size; i++)
        {
            if (bitmap[i])
                count += Rd[i];
        }
    }
    else
    {
        sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Ra, Ra, &condition);
        sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rb, Rb, &condition);
        sel_lt_T_bmp_independent_val_non_branching(result_size, bitmap_Rc, Rc, &condition);
        for (i = 0; i != result_size; ++i)
        {
            if (bitmap_Ra[i] && bitmap_Rb[i] && bitmap_Rc[i])
                bitmap[i] = 1;
            else
                bitmap[i] = 0;
        }
        for (i = 0; i < result_size; i++)
        {
            if (bitmap[i])
                count += Rd[i];
        }
    }
    return count;
}
/**
 * @brief perform select using Culomn-wise query processing model with independent bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx selalgo_cwm_bmp_independent_simd(idx condition, const idx &size_R, //size_R : the number of one group
                                const T *Ra,
                                const T *Rb,
                                const T *Rc,
                                const T *Rd,
                                std::vector<__mmask16> &bitmap_Ra,
                                std::vector<__mmask16> &bitmap_Rb,
                                std::vector<__mmask16> &bitmap_Rc,
                                std::vector<__mmask16> &bitmap)
{
    idx count = 0;
    idx i;
    idx result_size = size_R/16;

    sel_lt_T_bmp_independent_val_non_branching_simd(result_size, bitmap_Ra, Ra, &condition);
    sel_lt_T_bmp_independent_val_non_branching_simd(result_size, bitmap_Rb, Rb, &condition);
    sel_lt_T_bmp_independent_val_non_branching_simd(result_size, bitmap_Rc, Rc, &condition);
    
    __mmask16 tmp;
    for (i = 0; i != result_size; ++i)
    {   
        tmp = _kand_mask16(bitmap_Ra[i],bitmap_Rb[i]);
        bitmap[i] = _kand_mask16(tmp, bitmap_Rc[i]);        
    }
    for (i = 0; i != result_size; i++)
    {
        for(int k=0;k<16;k++){
            if (bitmap[i] & (1<<k)){
                count += 1;
            }
        }
    }
    
    return count;
}
/**
 * @brief case test using combined column-wise model with dynamic selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param sd_sv the shared dynamic selection vector
 * @return int the count of selection result
 */
idx casetest_combined_cwm_dsv_shared(idx condition, const idx &size_R,
                                     const T *Ra,
                                     const T *Rb,
                                     const T *Rc,
                                     const T *Rd,
                                     std::vector<T> &sd_sv,
                                     const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    idx current_size_ra = 0;
    idx current_size_rb = 0;
    idx current_size_rc = 0;
    if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition))
            {
                sd_sv.emplace_back(i);
            }
        }
        for (i = 0; i < sd_sv.size(); i++)
        {
            count += Rd[sd_sv[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition))
            {
                sd_sv.emplace_back(i);
            }
        }
        current_size_rc = sel_lt_T_dsel_shared_val_non_branching(sd_sv.size(), sd_sv, Rc, &condition);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sd_sv[i]];
        }
    }
    return count;
}
/**
 * @brief case test using combined column-wise model with independent selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx casetest_combined_cwm_dsv_independent(idx condition, const idx &size_R,
                                          const T *Ra,
                                          const T *Rb,
                                          const T *Rc,
                                          const T *Rd,
                                          std::vector<T> &d_sv_ra,
                                          std::vector<T> &d_sv_rb,
                                          std::vector<T> &d_sv_rc,
                                          const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;

    if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition))
            {
                d_sv_rb.emplace_back(i);
            }
        }
        sel_lt_T_dsel_independent_val_non_branching(d_sv_rb.size(), d_sv_rc, Rc, &condition, d_sv_rb);
        for (i = 0; i < d_sv_rc.size(); i++)
        {
            count += Rd[d_sv_rc[i]];
        }
    }
    return count;
}
/**
 * @brief case test using combined column-wise model with shared fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param sf_sv the shared fixed selection vector
 * @return int the count of selection result
 */
idx casetest_combined_cwm_fsv_shared(idx condition, const idx &size_R,
                                     const T *Ra,
                                     const T *Rb,
                                     const T *Rc,
                                     const T *Rd,
                                     idx *sf_sv,
                                     const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    idx current_size_ra = 0;
    idx current_size_rb = 0;
    idx current_size_rc = 0;
    if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition))
            {
                sf_sv[current_size_rc++] = i;
            }
        }
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition))
            {
                sf_sv[current_size_rc] = i;
                current_size_rc += (Rc[i] <= condition);
            }
        }
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE)
    {
        for (i = 0; i < result_size; i++)
        {
            if (Ra[i] <= condition)
            {
                sf_sv[current_size_rc] = i;
                current_size_rc += ((Rb[i] <= condition) && (Rc[i] <= condition));
            }
        }
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    else
    {
        for (i = 0; i < result_size; i++)
        {
            sf_sv[current_size_rc] = i;
            current_size_rc += ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition));
        }
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[sf_sv[i]];
        }
    }
    return count;
}
/**
 * @brief case test using combined column-wise model with independent fixed selection vector
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx casetest_combined_cwm_fsv_independent(idx condition, const idx &size_R,
                                          const T *Ra,
                                          const T *Rb,
                                          const T *Rc,
                                          const T *Rd,
                                          int *f_sv_ra,
                                          int *f_sv_rb,
                                          int *f_sv_rc,
                                          const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    idx current_size_ra = 0;
    idx current_size_rb = 0;
    idx current_size_rc = 0;
    if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition))
            {
                f_sv_rb[current_size_rb++] = i;
            }
        }
        current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, Rc, &condition, f_sv_rb);
        for (i = 0; i < current_size_rc; i++)
        {
            count += Rd[f_sv_rb[i]];
        }
    }
    return count;
}
/**
 * @brief case test using combined column-wise model with shared bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param s_bitmap the shared bitmap
 * @return int the count of selection result
 */
idx casetest_combined_cwm_bmp_shared(idx condition, const idx &size_R,
                                     const T *Ra,
                                     const T *Rb,
                                     const T *Rc,
                                     const T *Rd,
                                     std::vector<bool> &s_bitmap,
                                     const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO_THREE)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition) && (Rc[i] <= condition))
            {
                s_bitmap[i] = 1;
            }
            else
            {
                s_bitmap[i] = 0;
            }
        }
        for (i = 0; i < result_size; i++)
        {
            if (s_bitmap[i])
                count += Rd[i];
        }
    }
    else if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition))
            {
                s_bitmap[i] = 1;
            }
            else
            {
                s_bitmap[i] = 0;
            }
        }
        sel_lt_T_bmp_shared_val_non_branching(result_size, s_bitmap, Rc, &condition, false);
        for (i = 0; i < result_size; i++)
        {
            if (s_bitmap[i])
                count += Rd[i];
        }
    }
    return count;
}
/**
 * @brief case test using combined column-wise model with independent bitmap
 * @param condition determine the selection rate
 * @param size_R the number of column tuples
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @return int the count of selection result
 */
idx casetest_combined_cwm_bmp_independent(idx condition, const idx &size_R,
                                          const T *Ra,
                                          const T *Rb,
                                          const T *Rc,
                                          const T *Rd,
                                          std::vector<bool> &bitmap_Ra,
                                          std::vector<bool> &bitmap_Rb,
                                          std::vector<bool> &bitmap_Rc,
                                          std::vector<bool> &bitmap,
                                          const Selalgo_Branch &selalgo_branch)
{
    idx count = 0;
    idx i;
    idx result_size = size_R;
    if (selalgo_branch == Selalgo_Branch::BRANCH_ONE_TWO)
    {
        for (i = 0; i < result_size; i++)
        {
            if ((Ra[i] <= condition) && (Rb[i] <= condition))
            {
                bitmap_Rb[i] = 1;
            }
            else
            {
                bitmap_Rb[i] = 0;
            }
        }
        sel_lt_T_bmp_independent_val_branching(result_size, bitmap_Rc, Rc, &condition);
        for (i = 0; i != result_size; ++i)
        {
            if (bitmap_Rb[i] && bitmap_Rc[i])
                bitmap[i] = 1;
            else
                bitmap[i] = 0;
        }
        for (i = 0; i < result_size; i++)
        {
            if (bitmap[i])
                count += Rd[i];
        }
    }
    return count;
}
/**
 * @brief row-wise query processing model for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_rowwise_model(const idx &size_R,
                                const T *Ra, const T *Rb,
                                const T *Rc, const T *Rd,
                                const std::vector<idx> &conditions,
                                std::ofstream &selalgo_model_timefile,
                                std::ofstream &selalgo_model_lsr_timefile,
                                bool is_lsr)
{
    std::cout << ">>> Start test using row-wise model" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << pow((double)conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        gettimeofday(&start, NULL);
        count = selalgo_rowwise(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd);
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            selalgo_model_lsr_timefile << "BRANCH_ONE_TWO_THREE"
                                       << "\t"
                                       << "Row-wise query model"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Row-wise query model"
                                   << "\t"
                                   << "BRANCH_ONE_TWO_THREE"
                                   << "\t"
                                   << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
        }
    }
}
/**
 * @brief cloumn-wise query processing model with different dynamic vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_cwm_dsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)
{
    /*the shared dynamic selection vector*/
    std::cout << ">>> Start selection operator test using column-wise model with the shared dynamic vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        std::vector<int> sd_sv;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        count = selalgo_cwm_dsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sd_sv, selalgo_branch);
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Column-wise query model with the shared dynamic selection vector"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Column-wise query model with the shared dynamic selection vector"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Column-wise query model"
                             << "\t"
                             << "the shared dynamic selection vector"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     <<"Column-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Column-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Column-wise model with the shared dynamic selection vector and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Column-wise model with the shared dynamic selection vector and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
    /*the independent dynamic selection vector*/
    std::cout << ">>> Start selection operator test using column-wise model with the independent dynamic vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        std::vector<int> d_sv_ra;
        std::vector<int> d_sv_rb;
        std::vector<int> d_sv_rc;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        count = selalgo_cwm_dsv_independent(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, d_sv_ra, d_sv_rb, d_sv_rc, selalgo_branch);
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Column-wise query model with the independent dynamic selection vector"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Column-wise query model with the independent dynamic selection vector"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Column-wise query model"
                             << "\t"
                             << "the independent dynamic selection vector"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Column-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Column-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Column-wise model with the independent dynamic selection vector and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Column-wise model with the independent dynamic selection vector and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
}
/**
 * @brief vector-wise query processing model with different dynamic vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_dsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)
{
    /*the shared dynamic selection vector*/
    std::cout << ">>> Start selection operator test using vector-wise model with the shared dynamic vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        std::vector<int> sd_sv;
        sd_sv.reserve(size_v);
        idx vec_num = DATA_NUM / size_v;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_dsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v,
                                            sd_sv, selalgo_branch);
            sd_sv.clear();
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Vector-wise query model with the shared dynamic selection vector"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Vector-wise query model with the shared dynamic selection vector"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t"
                             << "the shared dynamic selection vector"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Vector-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Vector-wise model with the shared dynamic selection vector and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Vector-wise model with the shared dynamic selection vector and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Vector-wise model with the shared dynamic selection vector and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
    /*the independent dynamic selection vector*/
    std::cout << ">>> Start selection operator test using vector-wise model with the independent dynamic vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        std::vector<int> d_sv_ra;
        std::vector<int> d_sv_rb;
        std::vector<int> d_sv_rc;
        d_sv_ra.reserve(size_v);
        d_sv_rb.reserve(size_v);
        d_sv_rc.reserve(size_v);
        idx vec_num = DATA_NUM / size_v;
        //std::cout << vec_num << std::endl;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_dsv_independent(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v,
                                                 d_sv_ra, d_sv_rb, d_sv_rc, selalgo_branch);
            d_sv_ra.clear();
            d_sv_rb.clear();
            d_sv_rc.clear();
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Vector-wise query model with the independent dynamic selection vector"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Vector-wise query model with the independent dynamic selection vector"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
           
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t"
                             << "the independent dynamic selection vector"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Vector-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Vector-wise model with the independent dynamic selection vector and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Vector-wise model with the independent dynamic selection vector and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Vector-wise model with the independent dynamic selection vector and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
}
/**
 * @brief vector-wise query processing model with different dynamic vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_dsv_simd(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_timefile,
                          std::ofstream &selalgo_model_simd_timefile,
                          std::ofstream &selalgo_model_lsr_simd_timefile,
                          bool is_lsr)
{
    /*the shared dynamic selection vector*/
    std::cout << ">>> Start selection operator test using vector-wise model with the shared dynamic vector SIMD version" << std::endl;
    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;

        idx count = 0;
        timeval start, end;
        std::vector<T> sd_sv;
        sd_sv.reserve(size_v/16);    // size_v:1024 1024/16=64    sd_sv:64 so SIMD version can deal with 1024 once
        idx vec_num = DATA_NUM / (size_v/16);
        
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_dsv_shared_simd(conditions[select_idx], size_v/16, Ra + i * size_v/16, Rb + i * size_v/16, Rc + i * size_v/16, Rd + i * size_v/16,
                                            sd_sv);
            sd_sv.clear();
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            selalgo_model_lsr_simd_timefile << "SIMD_NON_BRANCH"
                                       << "\t";
            
            selalgo_model_lsr_simd_timefile << "Vector-wise query model with the shared dynamic selection vector SIMD version"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_simd_timefile << "Vector-wise query model with the shared dynamic selection vector SIMD version"
                                   << "\t";
            selalgo_model_simd_timefile  << "SIMD"
                                    << "\t";
            selalgo_model_simd_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t"
                             << "the shared dynamic selection vector"
                             << "\t";
            selalgo_timefile << "SIMD_NON_BRANCH"
                                     << "\t"
                                     << "Vector-wise model with the shared dynamic selection vector and NON_BRANCH"
                                     << "\t";
            selalgo_timefile << "Vector-wise model with SIMD_NON_BRANCH"
                                     << "\t";
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }

    /*the independent dynamic selection vector*/
    std::cout << ">>> Start selection operator test using vector-wise model with the independent dynamic vector SIMD version" << std::endl;
    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        std::vector<int> d_sv_ra;
        std::vector<int> d_sv_rb;
        std::vector<int> d_sv_rc;
        d_sv_ra.reserve(size_v);
        d_sv_rb.reserve(size_v);
        d_sv_rc.reserve(size_v);
        idx vec_num = DATA_NUM / size_v;
        //std::cout << vec_num << std::endl;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_dsv_independent_simd(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v,
                                                 d_sv_ra, d_sv_rb, d_sv_rc);
            d_sv_ra.clear();
            d_sv_rb.clear();
            d_sv_rc.clear();
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            selalgo_model_lsr_simd_timefile << "SIMD"
                                       << "\t";
            
            selalgo_model_lsr_simd_timefile << "Vector-wise query model with the independent dynamic selection vector SIMD version"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_simd_timefile << "Vector-wise query model with the independent dynamic selection vector SIMD version"
                                   << "\t";

            selalgo_model_simd_timefile << "SIMD"
                                   << "\t";
           
            selalgo_model_simd_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t"
                             << "the independent dynamic selection vector"
                             << "\t";

            selalgo_timefile << "SIMD"
                             << "\t"
                             << "Vector-wise model with the independent dynamic selection vector and SIMD"
                             << "\t";
            selalgo_timefile << "Vector-wise model with SIMD"
                             << "\t";
            
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }

    /*the independent dynamic selection vector gather*/
    std::cout << ">>> Start selection operator test using vector-wise model with SIMD gather version" << std::endl;
    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        
        idx vec_num = DATA_NUM / size_v;
        //std::cout << vec_num << std::endl;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_dsv_independent_simd_gather(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v);
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            selalgo_model_lsr_simd_timefile << "SIMD"
                                       << "\t";
            
            selalgo_model_lsr_simd_timefile << "Vector-wise query model with the independent dynamic selection vector SIMD version"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_simd_timefile << "Vector-wise query model with the independent dynamic selection vector SIMD version"
                                   << "\t";

            selalgo_model_simd_timefile << "SIMD"
                                   << "\t";
           
            selalgo_model_simd_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t"
                             << "the independent dynamic selection vector"
                             << "\t";

            selalgo_timefile << "SIMD"
                             << "\t"
                             << "Vector-wise model with the independent dynamic selection vector and SIMD"
                             << "\t";
            selalgo_timefile << "Vector-wise model with SIMD"
                             << "\t";
            
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
}

/**
 * @brief cloumn-wise query processing model with different fixed vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_cwm_fsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          std::ofstream &casestudy_timefile,
                          std::ofstream &casestudy_lsr_timefile,
                          bool is_lsr)
{
    /*the shared fixed selection vector*/
    std::cout << ">>> Start selection operator test using column-wise model with the shared fixed vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        const idx vector_size = size_R;
        int *sf_sv = new int[vector_size];
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        count = selalgo_cwm_fsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sf_sv, selalgo_branch);
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
           
            selalgo_model_lsr_timefile << "Column-wise query model with the shared fixed selection vector"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
            casestudy_lsr_timefile << "multipass processing mode"
                                   << "\t"
                                   << "Column-wise query model"
                                   << "\t";

            log_switch_branch(selalgo_branch, casestudy_lsr_timefile);
            
            casestudy_lsr_timefile << "multipass processing mode with Column-wise query model"
                                   << "\t"
                                   << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                   << "\t"
                                   << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Column-wise query model with the shared fixed selection vector"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Column-wise query model"
                             << "\t"
                             << "the shared fixed selection vector"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Column-wise model with the shared fixed selection vector and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Column-wise model with the shared fixed selection vector and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Column-wise model with the shared fixed selection vector and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Column-wise model with the shared fixed selection vector and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
            casestudy_timefile << "multipass processing mode"
                               << "\t"
                               << "Column-wise query model"
                               << "\t";

            log_switch_branch(selalgo_branch, casestudy_timefile);
           
            casestudy_timefile << "multipass processing mode with Column-wise query model"
                               << "\t"
                               << 0.1 * (select_idx + 1)
                               << "\t"
                               << ms << std::endl;
        }
    }

    /*the independent fixed selection vector*/
    std::cout << ">>> Start selection operator test using column-wise model with the independent fixed vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        const idx vector_size = size_R;
        int *f_sv_ra = new int[vector_size];
        int *f_sv_rb = new int[vector_size];
        int *f_sv_rc = new int[vector_size];
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        count = selalgo_cwm_fsv_independent(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, f_sv_ra, f_sv_rb, f_sv_rc, selalgo_branch);
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Column-wise query model with the independent fixed selection vector"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Column-wise query model with the independent fixed selection vector"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Column-wise query model"
                             << "\t"
                             << "the independent fixed selection vector"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Column-wise model with the independent fixed selection vector and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Column-wise model with the independent fixed selection vector and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Column-wise model with the independent fixed selection vector and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Column-wise model with the independent fixed selection vector and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
}
/**
 * @brief vector-wise query processing model with different fixed vector for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_fsv(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          std::ofstream &casestudy_timefile,
                          std::ofstream &casestudy_lsr_timefile,
                          bool is_lsr)
{
    /*the shared fixed selection vector*/
    std::cout << ">>> Start selection operator test using vector-wise model with the shared fixed vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;

        int *sf_sv = new int[size_v];
        idx vec_num = DATA_NUM / size_v;

        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_fsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, sf_sv, selalgo_branch);
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Vector-wise query model with the shared fixed selection vector"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
            casestudy_lsr_timefile << "multipass processing mode"
                                   << "\t"
                                   << "Vector-wise query model"
                                   << "\t";

            log_switch_branch(selalgo_branch, casestudy_lsr_timefile);
            
            casestudy_lsr_timefile << "multipass processing mode with Vector-wise query model"
                                   << "\t"
                                   << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                   << "\t"
                                   << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Vector-wise query model with the shared fixed selection vector"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
           
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t"
                             << "the shared fixed selection vector"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Vector-wise model with the shared fixed selection vector and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Vector-wise model with the shared fixed selection vector and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Vector-wise model with the shared fixed selection vector and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Vector-wise model with the shared fixed selection vector and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
            casestudy_timefile << "multipass processing mode"
                               << "\t"
                               << "Vector-wise query model"
                               << "\t";

            log_switch_branch(selalgo_branch, casestudy_timefile);
            
            casestudy_timefile << "multipass processing mode with Vector-wise query model"
                               << "\t"
                               << 0.1 * (select_idx + 1)
                               << "\t"
                               << ms << std::endl;
        }
    }

    // /*the independent fixed selection vector*/
    // std::cout << ">>> Start selection operator test using vector-wise model with the independent fixed vector" << std::endl;
    // for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    // {
    //     if (is_lsr)
    //         std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
    //     else
    //         std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
    //     idx count = 0;
    //     timeval start, end;
    //     const idx vector_size = size_R;
    //     int *f_sv_ra = new int[size_v];
    //     int *f_sv_rb = new int[size_v];
    //     int *f_sv_rc = new int[size_v];
    //     idx vec_num = DATA_NUM / size_v;
    //     // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
    //     gettimeofday(&start, NULL);
    //     for (idx i = 0; i != vec_num; ++i)
    //     {
    //         count += selalgo_cwm_fsv_independent(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, f_sv_ra, f_sv_rb, f_sv_rc, selalgo_branch);
    //     }
    //     gettimeofday(&end, NULL);
    //     double ms = calc_ms(end, start);
    //     if (is_lsr)
    //         std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    //     else
    //         std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    //     std::cout << "          Time: " << ms << "ms" << std::endl;
    //     if (is_lsr)
    //     {
    //         log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
    //         selalgo_model_lsr_timefile << "Vector-wise query model with the independent fixed selection vector"
    //                                    << "\t"
    //                                    << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
    //                                    << "\t"
    //                                    << ms << std::endl;
    //     }
    //     else
    //     {
    //         selalgo_model_timefile << "Vector-wise query model with the independent fixed selection vector"
    //                                << "\t";

    //         log_switch_branch(selalgo_branch, selalgo_model_timefile);
           
    //         selalgo_model_timefile << 0.1 * (select_idx + 1)
    //                                << "\t"
    //                                << ms << std::endl;
    //         selalgo_timefile << "Vector-wise query model"
    //                          << "\t"
    //                          << "the independent fixed selection vector"
    //                          << "\t";
    //         switch ((int)selalgo_branch)
    //         {
    //             case 0:
    //                 selalgo_timefile << "BRANCH_ONE_TWO_THREE"
    //                                  << "\t"
    //                                  << "Vector-wise model with the independent fixed selection vector and BRANCH_ONE_TWO_THREE"
    //                                  << "\t";
    //                 selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
    //                                  << "\t";
    //                 break;
    //             case 1:
    //                 selalgo_timefile << "BRANCH_ONE_TWO"
    //                                  << "\t"
    //                                  << "Vector-wise model with the independent fixed selection vector and BRANCH_ONE_TWO"
    //                                  << "\t";
    //                 selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
    //                                  << "\t";
    //             case 2:
    //                 selalgo_timefile << "BRANCH_ONE"
    //                                  << "\t"
    //                                  << "Vector-wise model with the independent fixed selection vector and BRANCH_ONE"
    //                                  << "\t";
    //                 selalgo_timefile << "Vector-wise model with BRANCH_ONE"
    //                                  << "\t";
    //             case 3:
    //                 selalgo_timefile << "NON_BRANCH"
    //                                  << "\t"
    //                                  << "Vector-wise model with the independent fixed selection vector and NON_BRANCH"
    //                                  << "\t";
    //                 selalgo_timefile << "Vector-wise model with NON_BRANCH"
    //                                  << "\t";
    //             default:
    //                 break;
    //         }
    //         selalgo_timefile << 0.1 * (select_idx + 1)
    //                          << "\t"
    //                          << ms << std::endl;
    //     }
    // }
}
/**
 * @brief cloumn-wise query processing model with different bitmap for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_cwm_bmp(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)

{
    /*the shared bitmap*/
    std::cout << ">>> Start selection operator test using column-wise model with the shared bitmap" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        std::vector<bool> bitmap;
        bitmap.reserve(DATA_NUM);
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        gettimeofday(&start, NULL);
        count = selalgo_cwm_bmp_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, bitmap, selalgo_branch);

        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        // time_results.emplace_back(ms);
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Column-wise query model with the shared bitmap"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Column-wise query model with the shared bitmap"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Column-wise query model"
                             << "\t"
                             << "the shared bitmap"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Column-wise model with the shared bitmap and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Column-wise model with the shared bitmap and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Column-wise model with the shared bitmap and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Column-wise model with the shared bitmap and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
    /*the independent bitmap*/
    std::cout << ">>> Start selection operator test using column-wise model with the independent bitmap" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        std::vector<bool> bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap;
        bitmap_Ra.reserve(DATA_NUM);
        bitmap_Rb.reserve(DATA_NUM);
        bitmap_Rc.reserve(DATA_NUM);
        bitmap.reserve(DATA_NUM);
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        gettimeofday(&start, NULL);
        count = selalgo_cwm_bmp_independent(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap, selalgo_branch);

        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        // time_results.emplace_back(ms);
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
            
            selalgo_model_lsr_timefile << "Column-wise query model with the independent bitmap"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Column-wise query model with the independent bitmap"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Column-wise query model"
                             << "\t"
                             << "the independent bitmap"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "BRANCH_ONE_TWO_THREE"
                                     << "\t"
                                     << "Column-wise model with the independent bitmap and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "BRANCH_ONE_TWO"
                                     << "\t"
                                     << "Column-wise model with the independent bitmap and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "BRANCH_ONE"
                                     << "\t"
                                     << "Column-wise model with the independent bitmap and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "NON_BRANCH"
                                     << "\t"
                                     << "Column-wise model with the independent bitmap and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Column-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
}
/**
 * @brief vector-wise query processing model with different bitmap for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_bmp(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_model_timefile,
                          std::ofstream &selalgo_model_lsr_timefile,
                          std::ofstream &selalgo_timefile,
                          bool is_lsr)

{
    std::cout << ">>> Start selection operator test using vector-wise model with the shared bitmap" << std::endl;
    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        std::vector<bool> bitmap;
        bitmap.reserve(size_v);
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        idx vec_num = DATA_NUM / size_v;
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_bmp_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, bitmap, selalgo_branch);
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        // time_results.emplace_back(ms);
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
           
            selalgo_model_lsr_timefile << "Vector-wise query model with the shared bitmap"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Vector-wise query model with the shared bitmap"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
            
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "Vector-wise model with the shared bitmap and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "Vector-wise model with the shared bitmap and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "Vector-wise model with the shared bitmap and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "Vector-wise model with the shared bitmap and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }

    std::cout << ">>> Start selection operator test using vector-wise model with the independent bitmap" << std::endl;
    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        std::vector<bool> bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap;
        bitmap_Ra.reserve(size_v);
        bitmap_Rb.reserve(size_v);
        bitmap_Rc.reserve(size_v);
        bitmap.reserve(size_v);
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        idx vec_num = DATA_NUM / size_v;
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_bmp_independent(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v,
                                                 bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap, selalgo_branch);
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        // time_results.emplace_back(ms);
        if (is_lsr)
        {
            log_switch_branch(selalgo_branch, selalgo_model_lsr_timefile);
       
            selalgo_model_lsr_timefile << "Vector-wise query model with the independent bitmap"
                                       << "\t"
                                       << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                       << "\t"
                                       << ms << std::endl;
        }
        else
        {
            selalgo_model_timefile << "Vector-wise query model with the independent bitmap"
                                   << "\t";

            log_switch_branch(selalgo_branch, selalgo_model_timefile);
           
            selalgo_model_timefile << 0.1 * (select_idx + 1)
                                   << "\t"
                                   << ms << std::endl;
            selalgo_timefile << "Vector-wise query model"
                             << "\t";
            switch ((int)selalgo_branch)
            {
                case 0:
                    selalgo_timefile << "Vector-wise model with the independent bitmap and BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO_THREE"
                                     << "\t";
                    break;
                case 1:
                    selalgo_timefile << "Vector-wise model with the independent bitmap and BRANCH_ONE_TWO"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE_TWO"
                                     << "\t";
                case 2:
                    selalgo_timefile << "Vector-wise model with the independent bitmap and BRANCH_ONE"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with BRANCH_ONE"
                                     << "\t";
                case 3:
                    selalgo_timefile << "Vector-wise model with the independent bitmap and NON_BRANCH"
                                     << "\t";
                    selalgo_timefile << "Vector-wise model with NON_BRANCH"
                                     << "\t";
                default:
                    break;
            }
            selalgo_timefile << 0.1 * (select_idx + 1)
                             << "\t"
                             << ms << std::endl;
        }
    }
}
/**
 * @brief vector-wise query processing model with different bitmap for selection algorithm implementation and testing SIMD version
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vwm_bmp_simd(const idx &size_R,
                          const T *Ra, const T *Rb,
                          const T *Rc, const T *Rd,
                          const std::vector<idx> &conditions,    // selection rate
                          const Selalgo_Branch &selalgo_branch,
                          std::ofstream &selalgo_timefile,
                          std::ofstream &selalgo_model_simd_timefile,
                          std::ofstream &selalgo_model_lsr_simd_timefile,
                          bool is_lsr)

{
    /* vector-wise model with the shared bitmap SIMD version */
    std::cout << ">>> Start selection operator test using vector-wise model with the shared bitmap SIMD version" << std::endl;
    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)  // selection rate from low to high
    {
        if (is_lsr)
           std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
           std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        
        idx count = 0;
        timeval start, end;
        idx vec_num = DATA_NUM / 16;  // deal with only one __m512i once

        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)  // the old src was "i != vec_num"
        {
            count += selalgo_cwm_bmp_shared_simd(conditions[select_idx], 16, Ra + i * 16, Rb + i * 16, Rc + i * 16, Rd + i * 16);
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);

        if (is_lsr)
           std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
           std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        // time_results.emplace_back(ms);
        if (is_lsr)
        {
           selalgo_model_lsr_simd_timefile << "Vector-wise query model with the shared bitmap SIMD version"
                                      << "\t"
                                      << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                      << "\t"
                                      << ms << std::endl;
        }
        else
        {
           selalgo_model_simd_timefile << "Vector-wise query model with the shared bitmap SIMD version"
                                  << "\t"
                                  << 0.1 * (select_idx + 1)
                                  << "\t"
                                  << ms << std::endl;
           selalgo_timefile << "Vector-wise query model SIMD version"
                            << "\t"
                            << "Vector-wise model with the shared bitmap and SIMD"
                            << "\t"
                            << "Vector-wise model with SIMD"
                            << "\t"
                            << 0.1 * (select_idx + 1)
                            << "\t"
                            << ms << std::endl;
        }
    }

    /* vector-wise model with the independent bitmap SIMD version */
   std::cout << ">>> Start selection operator test using vector-wise model with the independent bitmap SIMD version" << std::endl;
   for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
   {
       std::vector<__mmask16> bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap;
       bitmap_Ra.reserve(independent_bitmap_size_v);
       bitmap_Rb.reserve(independent_bitmap_size_v);
       bitmap_Rc.reserve(independent_bitmap_size_v);
       bitmap.reserve(independent_bitmap_size_v);

       std::fill(bitmap_Ra.begin(),bitmap_Ra.end(),0);

       if (is_lsr)
           std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
       else
           std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
       idx count = 0;
       timeval start, end;
       idx vec_num = DATA_NUM / independent_bitmap_size_v;  // deal one group(64 * 512b) once
       
       gettimeofday(&start, NULL);

       for (idx i = 0; i < vec_num; ++i)
       {
           count += selalgo_cwm_bmp_independent_simd(conditions[select_idx], independent_bitmap_size_v, Ra + i * independent_bitmap_size_v, Rb + i * independent_bitmap_size_v, Rc + i * independent_bitmap_size_v, Rd + i * independent_bitmap_size_v,
                                                bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap);
       }
       gettimeofday(&end, NULL);
       double ms = calc_ms(end, start);
       if (is_lsr)
           std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
       else
           std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
       std::cout << "          Time: " << ms << "ms" << std::endl;
    //    time_results.emplace_back(ms);
       if (is_lsr)
       {
           selalgo_model_lsr_simd_timefile << "Vector-wise query model with the independent bitmap SIMD version"
                                      << "\t"
                                      << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                      << "\t"
                                      << ms << std::endl;
       }
       else
       {
           selalgo_model_simd_timefile << "Vector-wise query model with the independent bitmap SIMD version"
                                  << "\t";
           selalgo_model_simd_timefile << "SIMD"
                                  << "\t";
           selalgo_model_simd_timefile << 0.1 * (select_idx + 1)
                                  << "\t"
                                  << ms << std::endl;

           selalgo_timefile << "Vector-wise query model SIMD version"
                            << "\t";
           selalgo_timefile << "Vector-wise model with the independent bitmap and SIMD"
                            << "\t";
           selalgo_timefile << "Vector-wise model with SIMD"
                            << "\t";
           selalgo_timefile << 0.1 * (select_idx + 1)
                            << "\t"
                            << ms << std::endl;
       }
   }

}
/**
 * @brief column-wise query processing model for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_columnwise_model(const idx &size_R,
                                   const T *Ra, const T *Rb,
                                   const T *Rc, const T *Rd,
                                   const std::vector<idx> &conditions,
                                   const Selalgo_Branch &selalgo_branch,
                                   std::ofstream &selalgo_model_timefile,
                                   std::ofstream &selalgo_model_lsr_timefile,
                                   std::ofstream &selalgo_timefile,
                                   std::ofstream &casestudy_timefile,
                                   std::ofstream &casestudy_lsr_timefile,
                                   bool is_lsr)
{
    /*column-wise query processing model with dynamic selection vector*/
    test_selalgo_cwm_dsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
    /*column-wise query processing model with fixed selection vector*/
    test_selalgo_cwm_fsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    /*column-wise query processing model with bitmap*/
    test_selalgo_cwm_bmp(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
}
/**
 * @brief vector-wise query processing model for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vectorwise_model(const idx &size_R,
                                   const T *Ra, const T *Rb,
                                   const T *Rc, const T *Rd,
                                   const std::vector<idx> &conditions,
                                   const Selalgo_Branch &selalgo_branch,
                                   std::ofstream &selalgo_model_timefile,
                                   std::ofstream &selalgo_model_lsr_timefile,
                                   std::ofstream &selalgo_timefile,
                                   std::ofstream &casestudy_timefile,
                                   std::ofstream &casestudy_lsr_timefile,
                                   bool is_lsr)
{
    // /*vector-wise query processing model with dynamic selection vector*/
    // test_selalgo_vwm_dsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
    /*vector-wise query processing model with fixed selection vector*/
    test_selalgo_vwm_fsv(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    // /*vector-wise query processing model with bitmap*/
    // test_selalgo_vwm_bmp(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, is_lsr);
}
/**
 * @brief vector-wise query processing model for selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo_vectorwise_model_simd(const idx &size_R,
                                   const T *Ra, const T *Rb,
                                   const T *Rc, const T *Rd,
                                   const std::vector<idx> &conditions,
                                   const Selalgo_Branch &selalgo_branch,
                                   std::ofstream &selalgo_timefile,
                                   std::ofstream &selalgo_model_simd_timefile,
                                   std::ofstream &selalgo_model_lsr_simd_timefile,
                                   bool is_lsr)
{
    /* vector-wise bitmap SIMD version */
    test_selalgo_vwm_bmp_simd(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_timefile, selalgo_model_simd_timefile, selalgo_model_lsr_simd_timefile, is_lsr);
    // /* vector-wise query processing model with dynamic selection vector */
    // test_selalgo_vwm_dsv_simd(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_branch, selalgo_timefile, selalgo_model_simd_timefile, selalgo_model_lsr_simd_timefile, is_lsr);
}
/**
 * @brief combined column-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_combined_columnwise_model(const idx &size_R,
                                         const T *Ra, const T *Rb,
                                         const T *Rc, const T *Rd,
                                         const std::vector<idx> &conditions,
                                         const Selalgo_Branch &selalgo_branch,
                                         std::ofstream &casestudy_timefile,
                                         std::ofstream &casestudy_lsr_timefile,
                                         bool is_lsr)
{
    /*the shared fixed selection vector*/
    std::cout << ">>> Start case test using combined column-wise model with the shared fixed vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;

        const idx vector_size = size_R;
        int *sf_sv = new int[vector_size];
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        count = casetest_combined_cwm_fsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sf_sv, selalgo_branch);
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            casestudy_lsr_timefile << "combined processing mode"
                                   << "\t"
                                   << "Column-wise query model"
                                   << "\t";

            log_switch_branch(selalgo_branch, casestudy_lsr_timefile);
          
            casestudy_lsr_timefile << "combined processing mode with Column-wise query model"
                                   << "\t"
                                   << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                   << "\t"
                                   << ms << std::endl;
        }
        else
        {
            casestudy_timefile << "combined processing mode"
                               << "\t"
                               << "Column-wise query model"
                               << "\t";

            log_switch_branch(selalgo_branch, casestudy_timefile);
          
            casestudy_timefile << "combined processing mode with Column-wise query model"
                               << "\t"
                               << 0.1 * (select_idx + 1)
                               << "\t"
                               << ms << std::endl;
        }
    }
}
/**
 * @brief multipass column-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_multipass_columnwise_model(const idx &size_R,
                                          const T *Ra, const T *Rb,
                                          const T *Rc, const T *Rd,
                                          const std::vector<idx> &conditions,
                                          const Selalgo_Branch &selalgo_branch,
                                          std::ofstream &casestudy_timefile,
                                          std::ofstream &casestudy_lsr_timefile,
                                          bool is_lsr)
{

    /*the shared fixed selection vector*/
    std::cout << ">>> Start case test using multipass column-wise model with the shared fixed vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;

        const idx vector_size = size_R;
        int *sf_sv = new int[vector_size];
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        count = selalgo_cwm_fsv_shared(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, sf_sv, selalgo_branch);
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;
        if (is_lsr)
        {
            casestudy_lsr_timefile << "multipass processing mode"
                                   << "\t"
                                   << "Column-wise query model"
                                   << "\t";

            log_switch_branch(selalgo_branch, casestudy_lsr_timefile);
         
            casestudy_lsr_timefile << "multipass processing mode with Column-wise query model"
                                   << "\t"
                                   << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                   << "\t"
                                   << ms << std::endl;
        }
        else
        {
            casestudy_timefile << "multipass processing mode"
                               << "\t"
                               << "Column-wise query model"
                               << "\t";

            log_switch_branch(selalgo_branch, casestudy_timefile);
           
            casestudy_timefile << "multipass processing mode with Column-wise query model"
                               << "\t"
                               << 0.1 * (select_idx + 1)
                               << "\t"
                               << ms << std::endl;
        }
    }
}
/**
 * @brief combined vector-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_combined_vectorwise_model(const idx &size_R,
                                         const T *Ra, const T *Rb,
                                         const T *Rc, const T *Rd,
                                         const std::vector<idx> &conditions,
                                         const Selalgo_Branch &selalgo_branch,
                                         std::ofstream &casestudy_timefile,
                                         std::ofstream &casestudy_lsr_timefile,
                                         bool is_lsr)
{

    /*the shared fixed selection vector*/
    std::cout << ">>> Start case test using combined vector-wise model with the shared fixed vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        int *sf_sv = new int[size_v];
        idx vec_num = DATA_NUM / size_v;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += casetest_combined_cwm_fsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, sf_sv, selalgo_branch);
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        std::cout << "          Time: " << ms << "ms" << std::endl;

        if (is_lsr)
        {
            casestudy_lsr_timefile << "combined processing mode"
                                   << "\t"
                                   << "Vector-wise query model"
                                   << "\t";

            log_switch_branch(selalgo_branch, casestudy_lsr_timefile);
            
            casestudy_lsr_timefile << "combined processing mode with Vector-wise query model"
                                   << "\t"
                                   << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                   << "\t"
                                   << ms << std::endl;
        }
        else
        {
            casestudy_timefile << "combined processing mode"
                               << "\t"
                               << "Vector-wise query model"
                               << "\t";

            log_switch_branch(selalgo_branch, casestudy_timefile);
            
            casestudy_timefile << "combined processing mode with Vector-wise query model"
                               << "\t"
                               << 0.1 * (select_idx + 1)
                               << "\t"
                               << ms << std::endl;
        }
    }
}
/**
 * @brief multipass vector-wise model for case test
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case_multipass_vectorwise_model(const idx &size_R,
                                          const T *Ra, const T *Rb,
                                          const T *Rc, const T *Rd,
                                          const std::vector<idx> &conditions,
                                          const Selalgo_Branch &selalgo_branch,
                                          std::ofstream &casestudy_timefile,
                                          std::ofstream &casestudy_lsr_timefile,
                                          bool is_lsr)
{

    /*the shared fixed selection vector*/
    std::cout << ">>> Start case test using multipass vector-wise model with the shared fixed vector" << std::endl;

    for (idx select_idx = 0; select_idx != conditions.size(); select_idx++)
    {
        if (is_lsr)
            std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << " %, total selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 3) * 100 << "%" << std::endl;
        else
            std::cout << "      column selection rate " << conditions[select_idx] << " %, total selection rate " << pow(conditions[select_idx], 3) / pow(100, 3) * 100 << "%" << std::endl;
        idx count = 0;
        timeval start, end;
        int *sf_sv = new int[size_v];
        idx vec_num = DATA_NUM / size_v;
        // count = dynamic_vector_col_branch(conditions[select_idx], DATA_NUM, Ra, Rb, Rc, Rd, ret1, ret2, ret3, branch);
        gettimeofday(&start, NULL);
        for (idx i = 0; i != vec_num; ++i)
        {
            count += selalgo_cwm_fsv_shared(conditions[select_idx], size_v, Ra + i * size_v, Rb + i * size_v, Rc + i * size_v, Rd + i * size_v, sf_sv, selalgo_branch);
        }
        gettimeofday(&end, NULL);
        double ms = calc_ms(end, start);
        if (is_lsr)
            std::cout << "          Result count of selection rate " << (double)pow(conditions[select_idx] / 10, 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
        else
            std::cout << "          Result count of selection rate " << pow(conditions[select_idx], 3) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;

        std::cout << "          Time: " << ms << "ms" << std::endl;

        if (is_lsr)
        {
        casestudy_lsr_timefile << "multipass processing mode"
                                << "\t"
                                << "Vector-wise query model"
                                << "\t";

        log_switch_branch(selalgo_branch,casestudy_lsr_timefile);
        
        casestudy_lsr_timefile << "multipass processing mode with Vector-wise query model"
                                << "\t"
                                << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                                << "\t"
                                << ms << std::endl;
        }
        else
        {
        casestudy_timefile << "multipass processing mode"
                            << "\t"
                            << "Vector-wise query model"
                            << "\t";

        log_switch_branch(selalgo_branch, casestudy_timefile);
        
        casestudy_timefile << "multipass processing mode with Vector-wise query model"
                            << "\t"
                            << 0.1 * (select_idx + 1)
                            << "\t"
                            << ms << std::endl;
        }
    }
}
/**
 * @brief selection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_selalgo(const idx &size_R,
                  const T *Ra, const T *Rb,
                  const T *Rc, const T *Rd,
                  const std::vector<idx> &conditions,
                  std::ofstream &selalgo_model_timefile,
                  std::ofstream &selalgo_model_lsr_timefile,
                  std::ofstream &selalgo_model_simd_timefile,
                  std::ofstream &selalgo_model_lsr_simd_timefile,
                  std::ofstream &selalgo_timefile,
                  std::ofstream &casestudy_timefile,
                  std::ofstream &casestudy_lsr_timefile,
                  bool is_lsr)
{
    /*1.Row-wise query processing model*/
    std::cout<<"Here is Row-wise query processing model\n";
    test_selalgo_rowwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_model_timefile, selalgo_model_lsr_timefile, is_lsr);

    // for (const auto branch : SELALGO_BRANCH)
    // {
    //     /*2. Column-wise query processing model*/
    //     std::cout<<"Here is Column-wise query processing model\n";
    //     test_selalgo_columnwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    //     /*3. Vector-wise query processing model*/
    //     std::cout<<"Here is Vector-wise query processing model\n";
    //     test_selalgo_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    // }
    // // std::cout<<"NON_BRANCH\n";
    // // test_selalgo_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, NON_BRANCH, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    // // std::cout<<"BRANCH_ONE_TWO_THREE\n";
    // // test_selalgo_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, BRANCH_ONE_TWO_THREE, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    // /*4. Vector-wise query processing model SIMD version*/
    // std::cout<<"Here is Vector-wise query processing model SIMD version\n";
    // test_selalgo_vectorwise_model_simd(DATA_NUM, Ra, Rb, Rc, Rd, conditions, NON_BRANCH, selalgo_timefile, selalgo_model_simd_timefile, selalgo_model_lsr_simd_timefile, is_lsr);
}
/**
 * @brief comparison test for selection algorithm cases
 * @param size_R
 * @param Ra
 * @param Rb
 * @param Rc
 * @param Rd
 * @param conditions
 * @param timefile
 * @return void
 */
void test_case(const idx &size_R,
               const T *Ra, const T *Rb,
               const T *Rc, const T *Rd,
               const std::vector<idx> &conditions,
               std::ofstream &casestudy_timefile,
               std::ofstream &casestudy_lsr_timefile,
               bool is_lsr)
{
    for (const auto branch : CASE_COMBINED_BRANCH)
    {
        /*1. combined column-wise processing model*/
        test_case_combined_columnwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
        /*2. combined vector-wise processing model*/
        test_case_combined_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    }
    for (const auto branch : CASE_MULTIPASS_BRANCH)
    {
        /*3. multi-pass column-wise query processing model*/
        test_case_multipass_columnwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
        /*4. multi-pass vector-wise query processing model*/
        test_case_multipass_vectorwise_model(DATA_NUM, Ra, Rb, Rc, Rd, conditions, branch, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    }
}

//pthread
static int node_mapping[MAX_NODES];
/**
 * Returns SMT aware logical to physical CPU mapping for a given thread id.
 */
int get_cpu_id(int thread_id) 
{
    if(!inited){
        int i;
        max_cpus  = sysconf(_SC_NPROCESSORS_ONLN);
        for(i = 0; i < max_cpus; i++){
            node_mapping[i] = i;
        }
        inited = 1;
    }

    return node_mapping[thread_id % max_cpus];
}
void * vector_ifsv_NONBranch_thread(void * param)
{
    int rv,start;
    int current_size_ra=0,current_size_rb=0,current_size_rc=0;
    // int count=0;
    int i=0;
    arg_bm * args = (arg_bm*) param;
    int vector_size = args->num_tuples;
    start = args->Ra.start;
    
    int *f_sv_ra = new int[vector_size]();
    int *f_sv_rb = new int[vector_size]();
    int *f_sv_rc = new int[vector_size]();
    

    #ifndef NO_TIMING
    /* the first thread checkpoints the start time */
    if(args->tid == 0){
        gettimeofday(&args->start, NULL);
        // startTimer(&args->timer1);
        // startTimer(&args->timer2); 
        // args->timer3 = 0; /* no partitionig phase */
    }
    #endif

    current_size_ra = sel_lt_T_fsel_independent_val_non_branching(vector_size, f_sv_ra, args->Ra.col+start, &args->condition, NULL);
    current_size_rb = sel_lt_T_fsel_independent_val_non_branching(current_size_ra, f_sv_rb,args->Rb.col+start, &args->condition, f_sv_ra);
    current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, args->Rc.col+start, &args->condition, f_sv_rb);
    
    args->num_results=current_size_rc;

    #ifndef NO_TIMING
    /* for a reliable timing we have to wait until all finishes */
    BARRIER_ARRIVE(args->barrier, rv);
    if(args->tid == 0){
      gettimeofday(&args->end, NULL);
    }
    #endif

    delete[] f_sv_ra;
    delete[] f_sv_rb;
    delete[] f_sv_rc;
}
void * vector_sb_SIMD_thread(void * param)
{
    int start,i,rv;
    arg_bm * args = (arg_bm*) param;
    int group_size = args->num_tuples/16;
    start = args->Ra.start;
    //SIMD_thread
     __mmask16 s_bitmap;
    idx result_size = 16;    // 16,一次一组

    #ifndef NO_TIMING
    /* the first thread checkpoints the start time */
    if(args->tid == 0){
        gettimeofday(&args->start, NULL);
    }
    #endif

    for(i=0;i<group_size;i++){
        sel_lt_T_bmp_shared_val_non_branching_simd(result_size, s_bitmap, args->Ra.col+start+i*16, &args->condition, true);
        sel_lt_T_bmp_shared_val_non_branching_simd(result_size, s_bitmap, args->Rb.col+start+i*16, &args->condition, false);
        sel_lt_T_bmp_shared_val_non_branching_simd(result_size, s_bitmap, args->Rc.col+start+i*16, &args->condition, false);

        int count_shared=0;
        for(int k=0;k<16;k++){
            if(s_bitmap & (1<<k)){
                count_shared+=1;
            }
        }
        args->num_results += count_shared;
    }

    #ifndef NO_TIMING
    /* for a reliable timing we have to wait until all finishes */
    BARRIER_ARRIVE(args->barrier, rv);
    if(args->tid == 0){
      gettimeofday(&args->end, NULL);
    }
    #endif
}
void * vector_ib_SIMD_thread(void * param)
{
    int start,i,rv;
    int count = 0;
    __mmask16 tmp;
    arg_bm * args = (arg_bm*) param;
    int num_tuple=args->num_tuples/1024;  //group
    start=args->Ra.start;
    std::vector<__mmask16> bitmap_Ra, bitmap_Rb, bitmap_Rc, bitmap;
    bitmap_Ra.reserve(64);
    bitmap_Rb.reserve(64);
    bitmap_Rc.reserve(64);
    bitmap.reserve(64);

    #ifndef NO_TIMING
    /* the first thread checkpoints the start time */
    if(args->tid == 0){
        gettimeofday(&args->start, NULL);
    }
    #endif

    for(i=0;i<num_tuple;i++){    // the count of tuple = 1024
        std::fill(bitmap_Ra.begin(),bitmap_Ra.end(),0);

        sel_lt_T_bmp_independent_val_non_branching_simd(64, bitmap_Ra, args->Ra.col+start+i*1024, &args->condition);
        sel_lt_T_bmp_independent_val_non_branching_simd(64, bitmap_Rb, args->Rb.col+start+i*1024, &args->condition);
        sel_lt_T_bmp_independent_val_non_branching_simd(64, bitmap_Rc, args->Rc.col+start+i*1024, &args->condition);
 
        for(int j=0;j<64;j++){
            tmp = _kand_mask16(bitmap_Ra[j],bitmap_Rb[j]);
            bitmap[j] = _kand_mask16(tmp, bitmap_Rc[j]); 

            for(int k=0;k<16;k++){
                if (bitmap[j] & (1<<k)){
                    count += 1;
                }
            }    
        }

    }

    args->num_results += count;

    #ifndef NO_TIMING
    /* for a reliable timing we have to wait until all finishes */
    BARRIER_ARRIVE(args->barrier, rv);
    if(args->tid == 0){
      gettimeofday(&args->end, NULL);
    }
    #endif
}
void vector_independent_fixed_selection_NONBRANCH_pthread_run(  const int &numR, 
                                                                const T *Ra, 
                                                                const T *Rb, 
                                                                const T *Rc, 
                                                                const T *Rd, 
                                                                int conditions, 
                                                                std::ofstream &selalgo_pthread_timefile,
                                                                std::ofstream &selalgo_pthread_lsr_timefile,
                                                                bool is_lsr,
                                                                int nthreads)
{
    // int numcondition=conditions.size();
    int32_t curR,numRthr; /* per thread num */
    int i, rv,result_count=0;
    cpu_set_t set;
    arg_bm args[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    curR=numR;
    numRthr = numR / nthreads;
    
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    if(rv != 0){
        printf("Couldn't create the barrier\n");
        exit(EXIT_FAILURE);
    }

    pthread_attr_init(&attr);

    for(i = 0; i < nthreads; i++){
        int cpu_idx = get_cpu_id(i);    //Returns SMT aware logical to physical CPU mapping for a given thread id.
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        // set args
        args[i].tid = i;
        args[i].barrier = &barrier;
        args[i].Ra.col=Ra;
        args[i].Rb.col=Rb;
        args[i].Rc.col=Rc;
        args[i].Rd.col=Rd;
        args[i].num_tuples = (i == (nthreads-1)) ? curR : numRthr;
            curR -= numRthr;
        args[i].Ra.start=i*numRthr;
        args[i].condition=conditions;

        // gettimeofday(&args[i].start,NULL);
        rv = pthread_create(&tid[i], &attr, vector_ifsv_NONBranch_thread, (void*)&args[i]);
        if (rv){
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
        // gettimeofday(&args[i].end,NULL);
    }
    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
        // gettimeofday(&args[i].end,NULL);
    }
    
    #ifndef NO_TIMING
    double ms = calc_ms(args[0].end, args[0].start);
    std::cout<<"\tusing time = "<<ms<<"ms\t";
    for(i = 0; i < nthreads; i++){
        result_count += args[i].num_results;
    }
    std::cout<<"result_count = "<<result_count<<"/"<<DATA_NUM<<std::endl;

    if(is_lsr){
        selalgo_pthread_lsr_timefile<< conditions << "\t"
                                    << nthreads << "\t"
                                    << "vector_independent_fixed_selection_NONBRANCH\t"
                                    << ms << "\t";
        switch (conditions)
        {
        case 1:
            selalgo_pthread_lsr_timefile<< 0.07/ms <<std::endl;
            break;
        case 2:
            selalgo_pthread_lsr_timefile<< 0.1034/ms <<std::endl;
            break;
        case 4:
            selalgo_pthread_lsr_timefile<< 0.1166/ms <<std::endl;
            break;
        case 10:
            selalgo_pthread_lsr_timefile<< 0.118/ms <<std::endl;
            break;
        case 21:
            selalgo_pthread_lsr_timefile<< 0.1026/ms <<std::endl;
            break;
        case 46:
            selalgo_pthread_lsr_timefile<< 0.1012/ms <<std::endl;
            break;
        
        default:
            break;
        }
    }
    else{
        selalgo_pthread_timefile<< conditions << "\t"
                                << nthreads << "\t"
                                << "vector_independent_fixed_selection_NONBRANCH\t"
                                << ms << "\t";
        switch (conditions)
        {
        case 46:
            selalgo_pthread_timefile<< 0.065/ms <<std::endl;
            break;
        case 58:
            selalgo_pthread_timefile<< 0.1024/ms <<std::endl;
            break;
        case 66:
            selalgo_pthread_timefile<< 0.1084/ms <<std::endl;
            break;
        case 73:
            selalgo_pthread_timefile<< 0.1004/ms <<std::endl;
            break;
        case 79:
            selalgo_pthread_timefile<< 0.098/ms <<std::endl;
            break;
        case 84:
            selalgo_pthread_timefile<< 0.1052/ms <<std::endl;
            break;
        case 88:
            selalgo_pthread_timefile<< 0.103/ms <<std::endl;
            break;
        case 92:
            selalgo_pthread_timefile<< 0.1186/ms <<std::endl;
            break;
        case 96:
            selalgo_pthread_timefile<< 0.0974/ms <<std::endl;
            break;
        case 100:
            selalgo_pthread_timefile<< 0.1028/ms <<std::endl;
            break;
        
        default:
            break;
        }
    }
    #endif
}
void vector_shared_bitmap_SIMD_pthread_run(  const int &numR, 
                                                                const T *Ra, 
                                                                const T *Rb, 
                                                                const T *Rc, 
                                                                const T *Rd, 
                                                                int conditions, 
                                                                std::ofstream &selalgo_pthread_timefile,
                                                                std::ofstream &selalgo_pthread_lsr_timefile,
                                                                bool is_lsr,
                                                                int nthreads)
{
    int32_t curR,numRthr; /* per thread num */
    int i, rv,result_count=0;
    cpu_set_t set;
    arg_bm args[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    curR=numR;
    numRthr = (numR / 16) / nthreads;    // one group for each thread, total numRthr groups
    // printf("numRthr=%d\n",numRthr);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    if(rv != 0){
        printf("Couldn't create the barrier\n");
        exit(EXIT_FAILURE);
    }

    pthread_attr_init(&attr);

    for(i = 0; i < nthreads; i++){
        int cpu_idx = get_cpu_id(i);    //Returns SMT aware logical to physical CPU mapping for a given thread id.
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        // set args
        args[i].num_results=0;
        args[i].tid = i;
        args[i].barrier = &barrier;
        args[i].Ra.col=Ra;
        args[i].Rb.col=Rb;
        args[i].Rc.col=Rc;
        args[i].Rd.col=Rd;
        args[i].num_tuples = (i == (nthreads-1)) ? curR : numRthr*16;
            curR -= numRthr*16;
        args[i].Ra.start=i*numRthr*16;
        args[i].condition=conditions;

        // gettimeofday(&args[i].start,NULL);
        /*vector_shared_bitmap_SIMD_pthread_run*/
        rv = pthread_create(&tid[i], &attr, vector_sb_SIMD_thread, (void*)&args[i]);
        if (rv){
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
        // gettimeofday(&args[i].end,NULL);
    }
    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
        // gettimeofday(&args[i].end,NULL);
    }

    #ifndef NO_TIMING
    double ms = calc_ms(args[0].end, args[0].start);
    std::cout<<"\tusing time = "<<ms<<"ms\t";
    for(i = 0; i < nthreads; i++){
        result_count += args[i].num_results;
    }
    std::cout<<"result_count = "<<result_count<<"/"<<DATA_NUM<<std::endl;

    if(is_lsr){
        selalgo_pthread_lsr_timefile << conditions << "\t"
                             << nthreads << "\t"
                             << "vector_shared_bitmap_SIMD\t"
                             << ms << "\t";

        switch (conditions)
        {
        case 1:
            selalgo_pthread_lsr_timefile<< 0.1104/ms <<std::endl;
            break;
        case 2:
            selalgo_pthread_lsr_timefile<< 0.0824/ms <<std::endl;
            break;
        case 4:
            selalgo_pthread_lsr_timefile<< 0.103/ms <<std::endl;
            break;
        case 10:
            selalgo_pthread_lsr_timefile<< 0.0912/ms <<std::endl;
            break;
        case 21:
            selalgo_pthread_lsr_timefile<< 0.0932/ms <<std::endl;
            break;
        case 46:
            selalgo_pthread_lsr_timefile<< 0.0948/ms <<std::endl;
            break;
        
        default:
            break;
        }
    }
    else{
        selalgo_pthread_timefile << conditions << "\t"
                             << nthreads << "\t"
                             << "vector_shared_bitmap_SIMD\t"
                             << ms << "\t";

        switch (conditions)
        {
        case 46:
            selalgo_pthread_timefile<< 0.1152/ms <<std::endl;
            break;
        case 58:
            selalgo_pthread_timefile<< 0.0958/ms <<std::endl;
            break;
        case 66:
            selalgo_pthread_timefile<< 0.103/ms <<std::endl;
            break;
        case 73:
            selalgo_pthread_timefile<< 0.0848/ms <<std::endl;
            break;
        case 79:
            selalgo_pthread_timefile<< 0.084/ms <<std::endl;
            break;
        case 84:
            selalgo_pthread_timefile<< 0.0816/ms <<std::endl;
            break;
        case 88:
            selalgo_pthread_timefile<< 0.0934/ms <<std::endl;
            break;
        case 92:
            selalgo_pthread_timefile<< 0.0874/ms <<std::endl;
            break;
        case 96:
            selalgo_pthread_timefile<< 0.0944/ms <<std::endl;
            break;
        case 100:
            selalgo_pthread_timefile<< 0.0958/ms <<std::endl;
            break;
        
        default:
            break;
        }
    }
    #endif

}
void vector_independent_bitmap_SIMD_pthread_run(    const int &numR, 
                                                    const T *Ra, 
                                                    const T *Rb, 
                                                    const T *Rc, 
                                                    const T *Rd, 
                                                    int conditions, 
                                                    std::ofstream &selalgo_pthread_timefile,
                                                    std::ofstream &selalgo_pthread_lsr_timefile,
                                                    bool is_lsr,
                                                    int nthreads)
{
    int32_t curR,numRthr; /* per thread num */
    int i, rv,result_count=0;
    cpu_set_t set;
    arg_bm args[nthreads];
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    curR=numR;
    numRthr = (numR/1024) / nthreads;    // one group for each thread, total numRthr groups

    // printf("numRthr=%d\n",numRthr);
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    if(rv != 0){
        printf("Couldn't create the barrier\n");
        exit(EXIT_FAILURE);
    }

    pthread_attr_init(&attr);

    for(i = 0; i < nthreads; i++){
        int cpu_idx = get_cpu_id(i);    //Returns SMT aware logical to physical CPU mapping for a given thread id.
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        // set args
        args[i].num_results=0;
        args[i].tid = i;
        args[i].barrier = &barrier;
        args[i].Ra.col=Ra;
        args[i].Rb.col=Rb;
        args[i].Rc.col=Rc;
        args[i].Rd.col=Rd;
        args[i].num_tuples = (i == (nthreads-1)) ? curR : numRthr*1024;
            curR -= numRthr*1024;
        args[i].Ra.start=i*numRthr*1024;
        args[i].condition=conditions;

        // gettimeofday(&args[i].start,NULL);
        /* vector_independent_bitmap_SIMD_pthread_run */
        rv = pthread_create(&tid[i], &attr, vector_ib_SIMD_thread, (void*)&args[i]);
        if (rv){
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
        // gettimeofday(&args[i].end,NULL);
    }
    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
        // gettimeofday(&args[i].end,NULL);
    }

    #ifndef NO_TIMING
    double ms = calc_ms(args[0].end, args[0].start);
    std::cout<<"\tusing time = "<<ms<<"ms\t";
    for(i = 0; i < nthreads; i++){
        result_count += args[i].num_results;
    }
    std::cout<<"result_count = "<<result_count<<"/"<<DATA_NUM<<std::endl;

    if(is_lsr){
        selalgo_pthread_lsr_timefile << conditions << "\t"
                             << nthreads << "\t"        
                             << "vector_independent_bitmap_SIMD\t"
                             << ms << "\t";

        switch (conditions)
        {
        case 1:
            selalgo_pthread_lsr_timefile<< 0.0908/ms <<std::endl;
            break;
        case 2:
            selalgo_pthread_lsr_timefile<< 0.1082/ms <<std::endl;
            break;
        case 4:
            selalgo_pthread_lsr_timefile<< 0.1224/ms <<std::endl;
            break;
        case 10:
            selalgo_pthread_lsr_timefile<< 0.0988/ms <<std::endl;
            break;
        case 21:
            selalgo_pthread_lsr_timefile<< 0.1062/ms <<std::endl;
            break;
        case 46:
            selalgo_pthread_lsr_timefile<< 0.1036/ms <<std::endl;
            break;
        
        default:
            break;
        }
    }
    else{
        selalgo_pthread_timefile << conditions << "\t"
                             << nthreads << "\t"        
                             << "vector_independent_bitmap_SIMD\t"
                             << ms << "\t";

        switch (conditions)
        {
        case 46:
            selalgo_pthread_timefile<< 0.1096/ms <<std::endl;
            break;
        case 58:
            selalgo_pthread_timefile<< 0.0976/ms <<std::endl;
            break;
        case 66:
            selalgo_pthread_timefile<< 0.0836/ms <<std::endl;
            break;
        case 73:
            selalgo_pthread_timefile<< 0.0844/ms <<std::endl;
            break;
        case 79:
            selalgo_pthread_timefile<< 0.1148/ms <<std::endl;
            break;
        case 84:
            selalgo_pthread_timefile<< 0.1044/ms <<std::endl;
            break;
        case 88:
            selalgo_pthread_timefile<< 0.095/ms <<std::endl;
            break;
        case 92:
            selalgo_pthread_timefile<< 0.0894/ms <<std::endl;
            break;
        case 96:
            selalgo_pthread_timefile<< 0.0874/ms <<std::endl;
            break;
        case 100:
            selalgo_pthread_timefile<< 0.0918/ms <<std::endl;
            break;
        
        default:
            break;
        }
    }
    #endif
}
void test_selalgo_pthread(  const int &numR, 
                            const T *Ra, 
                            const T *Rb, 
                            const T *Rc, 
                            const T *Rd, 
                            std::vector<int> conditions, 
                            std::ofstream &selalgo_pthread_timefile,
                            std::ofstream &selalgo_pthread_lsr_timefile,
                            bool is_lsr,
                            int nthreads)
{    
    int cur_condition;
    int numcondition=conditions.size();
    /* 1.Vector-wise the independent fixed selection vector NON_BRANCH */
    std::cout<<">>>Here is Vector-wise the independent fixed selection vector NON_BRANCH pthread version"
             << "\tnthreads=" <<nthreads
             <<std::endl;
    for(int i=0;i<numcondition;i++){    // for each condition
        cur_condition=conditions[i];
        std::cout<<"  condition = "<<cur_condition<<std::endl;
        vector_independent_fixed_selection_NONBRANCH_pthread_run(DATA_NUM,Ra,Rb,Rc,Rd,cur_condition,selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,is_lsr,nthreads);
    }

    /* 2.Vector-wise shared bitmap SIMD version */
    std::cout<<">>>Here is Vector-wise shared bitmap SIMD pthread version"
             << "\tnthreads=" <<nthreads
             <<std::endl;
    for(int i=0;i<numcondition;i++){    // for each condition
        cur_condition=conditions[i];
        std::cout<<"  condition = "<<cur_condition<<std::endl;
        vector_shared_bitmap_SIMD_pthread_run(DATA_NUM,Ra,Rb,Rc,Rd,cur_condition,selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,is_lsr,nthreads);
    }

    /* 3.Vector-wise independent bitmap SIMD version */
    std::cout<<">>>Here is Vector-wise independent bitmap SIMD pthread version"
             << "\tnthreads=" <<nthreads
             <<std::endl;
    for(int i=0;i<numcondition;i++){    // for each condition
        cur_condition=conditions[i];
        std::cout<<"  condition = "<<cur_condition<<std::endl;
        vector_independent_bitmap_SIMD_pthread_run(DATA_NUM,Ra,Rb,Rc,Rd,cur_condition,selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,is_lsr,nthreads);
    }
}


void *vector_ifsv_NONBranch_thread_numa(void *param){
    int rv,start;
    int current_size_ra=0,current_size_rb=0,current_size_rc=0;
    // int count=0;
    int i=0;
    arg_bm * args = (arg_bm*) param;
    int vector_size = args->num_tuples;
    start = args->Ra.start;
    
    // int *f_sv_ra = new int[vector_size]();
    // int *f_sv_rb = new int[vector_size]();
    // int *f_sv_rc = new int[vector_size]();
    

    #ifndef NO_TIMING
    /* the first thread checkpoints the start time */
    if(args->tid == 0){
        gettimeofday(&args->start, NULL);
        // startTimer(&args->timer1);
        // startTimer(&args->timer2); 
        // args->timer3 = 0; /* no partitionig phase */
    }
    #endif

    current_size_ra = sel_lt_T_fsel_independent_val_non_branching(vector_size, f_sv_ra, args->Ra.col+start, &args->condition, NULL);
    current_size_rb = sel_lt_T_fsel_independent_val_non_branching(current_size_ra, f_sv_rb,args->Rb.col+start, &args->condition, f_sv_ra);
    current_size_rc = sel_lt_T_fsel_independent_val_non_branching(current_size_rb, f_sv_rc, args->Rc.col+start, &args->condition, f_sv_rb);
    
    args->num_results=current_size_rc;

    #ifndef NO_TIMING
    /* for a reliable timing we have to wait until all finishes */
    BARRIER_ARRIVE(args->barrier, rv);
    if(args->tid == 0){
      gettimeofday(&args->end, NULL);
    }
    #endif

    // delete[] f_sv_ra;
    // delete[] f_sv_rb;
    // delete[] f_sv_rc;
}
void vector_independent_fixed_selection_NONBRANCH_pthread_run_numa(  
                                                                const int &numR, 
                                                                int **vec_a_p,
                                                                int **vec_b_p,
                                                                int **vec_c_p,
                                                                int **vec_d_p, 
                                                                int conditions, 
                                                                // std::ofstream &selalgo_pthread_timefile,
                                                                // std::ofstream &selalgo_pthread_lsr_timefile,
                                                                // bool is_lsr,
                                                                int nthreads)
{
    int32_t curR,numRthr; /* per thread num */
    int i, rv,result_count=0,numa_len;
    cpu_set_t set;
    arg_bm_numa args[nthreads];
    pthread_t tid[nthreads];
    int *f_sv_ra[nthreads];
    int *f_sv_rb[nthreads];
    int *f_sv_rc[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    curR=numR;
    numRthr = numR / nthreads;
    
    int numa_num = numa_max_node() + 1;
    
    for (i = 0; i < nthreads; i++)
    {
        int numa_id = get_numa_id(i);
        bind_numa(numa_id);
        numa_len=(i == (nthreads-1)) ? curR : numR/numa_num;
            curR -= numR/numa_num;
        /* numa_alloc for f_sv_ra, for f_sv_rb, for f_sv_rc */
        f_sv_ra[i] = (int *)numa_alloc(sizeof(int) * numa_len);
        memset(f_sv_ra[i], 0, curR * sizeof(int));
        f_sv_rb[i] = (int *)numa_alloc(sizeof(int) * numa_len);
        memset(f_sv_rb[i], 0, curR * sizeof(int));
        f_sv_rc[i] = (int *)numa_alloc(sizeof(int) * numa_len);
        memset(f_sv_rc[i], 0, curR * sizeof(int));
    }

    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    if(rv != 0){
        printf("Couldn't create the barrier\n");
        exit(EXIT_FAILURE);
    }

    pthread_attr_init(&attr);
    for(i = 0; i < nthreads; i++){
        int cpu_idx = get_cpu_id(i);    //Returns SMT aware logical to physical CPU mapping for a given thread id.
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);

        // set args
        args[i].tid = i;
        args[i].barrier = &barrier;
        args[i].vec_a_p=vec_a_p;
        args[i].vec_b_p=vec_b_p;
        args[i].vec_c_p=vec_c_p;
        args[i].vec_d_p=vec_d_p;
        args[i].num_tuples = numa_len;
        args[i].Ra.start=i*numRthr;
        args[i].condition=conditions;

        // gettimeofday(&args[i].start,NULL);
        rv = pthread_create(&tid[i], &attr, vector_ifsv_NONBranch_thread_numa, (void*)&args[i]);
        if (rv){
            printf("ERROR; return code from pthread_create() is %d\n", rv);
            exit(-1);
        }
        // gettimeofday(&args[i].end,NULL);
    }
    for(i = 0; i < nthreads; i++){
        pthread_join(tid[i], NULL);
        // gettimeofday(&args[i].end,NULL);
    }
    
    #ifndef NO_TIMING
    double ms = calc_ms(args[0].end, args[0].start);
    std::cout<<"\tusing time = "<<ms<<"ms\t";
    for(i = 0; i < nthreads; i++){
        result_count += args[i].num_results;
    }
    std::cout<<"result_count = "<<result_count<<"/"<<DATA_NUM<<std::endl;

    // if(is_lsr){
    //     selalgo_pthread_lsr_timefile<< conditions << "\t"
    //                                 << nthreads << "\t"
    //                                 << "vector_independent_fixed_selection_NONBRANCH\t"
    //                                 << ms << "\t";
    //     switch (conditions)
    //     {
    //     case 1:
    //         selalgo_pthread_lsr_timefile<< 0.07/ms <<std::endl;
    //         break;
    //     case 2:
    //         selalgo_pthread_lsr_timefile<< 0.1034/ms <<std::endl;
    //         break;
    //     case 4:
    //         selalgo_pthread_lsr_timefile<< 0.1166/ms <<std::endl;
    //         break;
    //     case 10:
    //         selalgo_pthread_lsr_timefile<< 0.118/ms <<std::endl;
    //         break;
    //     case 21:
    //         selalgo_pthread_lsr_timefile<< 0.1026/ms <<std::endl;
    //         break;
    //     case 46:
    //         selalgo_pthread_lsr_timefile<< 0.1012/ms <<std::endl;
    //         break;
    //     default:
    //         break;
    //     }
    // }
    // else{
    //     selalgo_pthread_timefile<< conditions << "\t"
    //                             << nthreads << "\t"
    //                             << "vector_independent_fixed_selection_NONBRANCH\t"
    //                             << ms << "\t";
    //     switch (conditions)
    //     {
    //     case 46:
    //         selalgo_pthread_timefile<< 0.065/ms <<std::endl;
    //         break;
    //     case 58:
    //         selalgo_pthread_timefile<< 0.1024/ms <<std::endl;
    //         break;
    //     case 66:
    //         selalgo_pthread_timefile<< 0.1084/ms <<std::endl;
    //         break;
    //     case 73:
    //         selalgo_pthread_timefile<< 0.1004/ms <<std::endl;
    //         break;
    //     case 79:
    //         selalgo_pthread_timefile<< 0.098/ms <<std::endl;
    //         break;
    //     case 84:
    //         selalgo_pthread_timefile<< 0.1052/ms <<std::endl;
    //         break;
    //     case 88:
    //         selalgo_pthread_timefile<< 0.103/ms <<std::endl;
    //         break;
    //     case 92:
    //         selalgo_pthread_timefile<< 0.1186/ms <<std::endl;
    //         break;
    //     case 96:
    //         selalgo_pthread_timefile<< 0.0974/ms <<std::endl;
    //         break;
    //     case 100:
    //         selalgo_pthread_timefile<< 0.1028/ms <<std::endl;
    //         break;   
    //     default:
    //         break;
    //     }
    // }
    #endif
}

void test_selalgo_numa( const idx& size_R,
                        const std::vector<idx> &conditions,
                        bool is_lsr)
{
    int nthreads_numa=2;  // the number of pthread
    if(numa_available() < 0) {
        printf("Your system does not support NUMA API\n");
        return;
    }
    // else{
    //     printf("numa_available=%d\n",numa_available());
    // }

    /* gendata_numa */ 
    int numa_num = numa_max_node() + 1;
    // printf("numa_num=%d\n",numa_num);
    int *vec_a_p[numa_num], *vec_b_p[numa_num], *vec_c_p[numa_num], *vec_d_p[numa_num];  
    get_numa_info();
    for (int i = 0; i < numa_num; i++)
    {
    bind_numa(i);
    vec_a_p[i] = (int *)numa_alloc(sizeof(int) * size_R/numa_num);
    vec_b_p[i] = (int *)numa_alloc(sizeof(int) * size_R/numa_num);
    vec_c_p[i] = (int *)numa_alloc(sizeof(int) * size_R/numa_num);
    vec_d_p[i] = (int *)numa_alloc(sizeof(int) * size_R/numa_num);
    }
    // generate data in dimvec_c[]
    gen_data(size_R, numa_num,vec_a_p,vec_b_p,vec_c_p,vec_d_p,is_lsr);
   
    /* numa test */ 
    int cur_condition;
    int numcondition=conditions.size();
    /* 1.Vector-wise the independent fixed selection vector NON_BRANCH */
    std::cout<<">>>Here is Vector-wise the independent fixed selection vector NON_BRANCH NUMA version"<<std::endl;
    for(int i=0;i<numcondition;i++){    // for each condition
        cur_condition=conditions[i];
        std::cout<<"  condition = "<<cur_condition<<std::endl;
        vector_independent_fixed_selection_NONBRANCH_pthread_run_numa(size_R,cur_condition,nthreads_numa,);
    }
    // /* 2.Vector-wise shared bitmap SIMD version */
    // std::cout<<">>>Here is Vector-wise shared bitmap SIMD NUMA version"<<std::endl;
    // for(int i=0;i<numcondition;i++){    // for each condition
    //     cur_condition=conditions[i];
    //     std::cout<<"  condition = "<<cur_condition<<std::endl;
    //     vector_shared_bitmap_SIMD_pthread_run_numa(DATA_NUM,Ra,Rb,Rc,Rd,cur_condition,selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,is_lsr,nthreads);
    // }
    // /* 3.Vector-wise independent bitmap SIMD version */
    // std::cout<<">>>Here is Vector-wise independent bitmap SIMD NUMA version"<<std::endl;
    // for(int i=0;i<numcondition;i++){    // for each condition
    //     cur_condition=conditions[i];
    //     std::cout<<"  condition = "<<cur_condition<<std::endl;
    //     vector_independent_bitmap_SIMD_pthread_run_numa(DATA_NUM,Ra,Rb,Rc,Rd,cur_condition,selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,is_lsr,nthreads);
    // }

    // test_OLAPcore_vectorwise_numa(cmd_params.sf, sele_array, dimvec_array_numa, bitmap_array, fk_array_numa,
    //                     M1_p, M2_p, factor, orders, dimvec_nums, group_nums, cmd_params.nthreads, cmd_params.sqlnum, timefile);

    numa_free(vec_a_p,sizeof(int) * size_R/numa_num);
    numa_free(vec_b_p,sizeof(int) * size_R/numa_num);
    numa_free(vec_c_p,sizeof(int) * size_R/numa_num);
    numa_free(vec_d_p,sizeof(int) * size_R/numa_num);

    // delete[] vec_a_p;
    // delete[] vec_b_p;
    // delete[] vec_c_p;
    // delete[] vec_d_p;
}

int main(int argc, char **argv)
{
    bool is_lsr = false;
    static int is_lsr_flag;
    int opt;
    static struct option long_options[] =
            {
                    /* These options set a flag. */
                    {"is_lsr", no_argument, &is_lsr_flag, 1},
                    {0, 0, 0, 0}};
    const char *optstring = "ab:nr:";
    int option_index = 0;
    opt = getopt_long(argc, argv, optstring, long_options,
                      &option_index);
    is_lsr = is_lsr_flag;

    std::ofstream   selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, 
                    casestudy_timefile, casestudy_lsr_timefile, 
                    selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,  
                    selalgo_model_simd_timefile, selalgo_model_lsr_simd_timefile;
    if (is_lsr)
    {
        selalgo_pthread_lsr_timefile.open(SELALGO_PTHREAD_LSR_TIMEFILE,std::ios::out | std::ios::trunc);    
        selalgo_model_lsr_timefile.open(SELALGO_MODEL_LSR_TIME_FILE, std::ios::out | std::ios::trunc);
        // casestudy_lsr_timefile.open(CASESTUDY_LSR_TIME_FILE, std::ios::out | std::ios::trunc);
        selalgo_model_lsr_simd_timefile.open(SELALGO_MODEL_LSR_SIMD_FILE, std::ios::out | std::ios::trunc);
    }
    else
    {
        selalgo_pthread_timefile.open(SELALGO_PTHREAD_TIMEFILE,std::ios::out | std::ios::trunc);    
        selalgo_model_timefile.open(SELALGO_MODEL_TIME_FILE, std::ios::out | std::ios::trunc);
        selalgo_timefile.open(SELALGO_TIME_FILE, std::ios::out | std::ios::trunc);
        // casestudy_timefile.open(CASESTUDY_TIME_FILE, std::ios::out | std::ios::trunc);
        selalgo_model_simd_timefile.open(SELALGO_MODEL_SIMD_FILE, std::ios::out | std::ios::trunc);
    }
    log_write_header(is_lsr, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile);

    // T *Ra = new T[DATA_NUM];
    // T *Rb = new T[DATA_NUM];
    // T *Rc = new T[DATA_NUM];
    // T *Rd = new T[DATA_NUM];
    std::vector<int> conditions;
    // gen_data(DATA_NUM, Ra, Rb, Rc, Rd, is_lsr);
    gen_conditions(conditions, is_lsr);
    // /*Selection algorithms for branching and non-branching  implementations and test*/
    // std::cout<<"Here are Selection algorithms for branching and non-branching  implementations and test\n";
    // test_selalgo(DATA_NUM, Ra, Rb, Rc, Rd, conditions, selalgo_model_timefile, selalgo_model_lsr_timefile, selalgo_model_simd_timefile, selalgo_model_lsr_simd_timefile, selalgo_timefile, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    // /*top3 of Selection algorithms pthread test*/
    // std::cout<<"Here are pthread tests for three algorithms\n";
    test_selalgo_pthread(DATA_NUM,Ra,Rb,Rc,Rd,conditions,selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,is_lsr,2);
    // test_selalgo_pthread(DATA_NUM,Ra,Rb,Rc,Rd,conditions,selalgo_pthread_timefile,selalgo_pthread_lsr_timefile,is_lsr,48);
    // /*Case study*/
    // // std::cout<<"Here is case-study test\n";
    // // test_case(DATA_NUM, Ra, Rb, Rc, Rd, conditions, casestudy_timefile, casestudy_lsr_timefile, is_lsr);
    // delete[] Ra;
    // delete[] Rb;
    // delete[] Rc;
    // delete[] Rd;

    /*NUMA*/
    std::cout<<"Here are NUMA test for three algorithms\n"; 
    test_selalgo_numa(DATA_NUM, conditions, is_lsr);

    conditions.clear();
    selalgo_model_timefile.close();
    selalgo_model_lsr_timefile.close();
    selalgo_timefile.close();
    selalgo_model_simd_timefile.close();
    selalgo_model_lsr_simd_timefile.close();
    // casestudy_timefile.close();
    // casestudy_lsr_timefile.close();
}
