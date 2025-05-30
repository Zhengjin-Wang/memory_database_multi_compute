/**
 * @file test_project_option.cpp
 * @author Han Ruichen (hanruichen@ruc.edu.cn)
 * @brief test of projection algorithms
 *
 * @version 0.1
 * @date 2023-04-12
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "../include/metadata.h"
#include <bitset>
#include "../include/gendata_util.hpp"
#include "../include/statistical_analysis_util.hpp"
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
row_store_min row_min[67108864];
row_store_max row_max[67108864];

/**
 * @brief projection calculation
 *        calculate a row per iteration
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param result
 * @return int the result count.
 */

 int proalgo_rowwise_simd(int condition, const idx &size_R,
  row_store_min *row_min,
  std::vector<std::pair<int, int>> &result)
{
int count = 0;
idx i = 0;
__m512i v_cond = _mm512_set1_epi32(condition);

// 假设row_store_min结构体是紧凑的，每个成员都是int类型
// 处理16个元素一组的数据
for (; i + 16 <= size_R; i += 16) {
// 加载Ra值
__m512i v_ra = _mm512_setr_epi32(
row_min[i].Ra, row_min[i+1].Ra, row_min[i+2].Ra, row_min[i+3].Ra,
row_min[i+4].Ra, row_min[i+5].Ra, row_min[i+6].Ra, row_min[i+7].Ra,
row_min[i+8].Ra, row_min[i+9].Ra, row_min[i+10].Ra, row_min[i+11].Ra,
row_min[i+12].Ra, row_min[i+13].Ra, row_min[i+14].Ra, row_min[i+15].Ra
);

// 加载Rc值
__m512i v_rc = _mm512_setr_epi32(
row_min[i].Rc, row_min[i+1].Rc, row_min[i+2].Rc, row_min[i+3].Rc,
row_min[i+4].Rc, row_min[i+5].Rc, row_min[i+6].Rc, row_min[i+7].Rc,
row_min[i+8].Rc, row_min[i+9].Rc, row_min[i+10].Rc, row_min[i+11].Rc,
row_min[i+12].Rc, row_min[i+13].Rc, row_min[i+14].Rc, row_min[i+15].Rc
);

// 比较Ra和Rc是否都小于等于condition
__mmask16 mask_ra = _mm512_cmple_epi32_mask(v_ra, v_cond);
__mmask16 mask_rc = _mm512_cmple_epi32_mask(v_rc, v_cond);
__mmask16 mask = _mm512_kand(mask_ra, mask_rc);

// 处理满足条件的元素
while (mask) {
int idx = _tzcnt_u32(mask);
result.emplace_back(row_min[i + idx].Ra, row_min[i + idx].Rc);
count++;
mask = mask & (mask-1);
}
}

// 处理剩余的元素
for (; i < size_R; ++i) {
if (row_min[i].Ra <= condition && row_min[i].Rc <= condition) {
count++;
result.emplace_back(row_min[i].Ra, row_min[i].Rc);
}
}

return count;
}

int proalgo_rowwise(int condition, const idx &size_R,
                    row_store_min *row_min,
                    std::vector<std::pair<int, int>> &result)
{
  // int count = 0;
  // idx i;
  // idx result_size = size_R;
  // for (i = 0; i != result_size; ++i)
  // {
  //   if (row_min[i].Ra <= condition && row_min[i].Rc <= condition)
  //   {
  //     count++;
  //     result.emplace_back(row_min[i].Ra, row_min[i].Rc);
  //   }
  // }
  int count = proalgo_rowwise_simd(condition, size_R, row_min, result);
  return count;
}
/**
 * @brief projection calculation
 *        calculate a row per iteration
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param result
 * @return int the result count.
 */
int proalgo_rowwise(int condition, const idx &size_R,
                    row_store_min *row_min,
                    struct fixed_arrays &result)
{
  int count = 0;
  idx i;
  idx result_size = size_R;
  for (i = 0; i != result_size; ++i)
  {
    if (row_min[i].Ra <= condition && row_min[i].Rc <= condition)
    {

      result.pos_value1[count] = row_min[i].Ra; // value 1
      result.value2[count] = row_min[i].Rc;
      count++;
    }
  }
  return count;
}
/**
 * @brief projection calculation
 *        calculate a row per iteration
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param result
 * @return int the result count.
 */
int proalgo_rowwise(int condition, const idx &size_R,
                    row_store_max *row_max,
                    std::vector<std::pair<int, int>> &result)
{
  int count = 0;
  idx i;
  idx result_size = size_R;
  for (i = 0; i != result_size; ++i)
  {
    if (row_max[i].Ra <= condition && row_max[i].Rc <= condition)
    {
      count++;
      result.emplace_back(row_max[i].Ra, row_max[i].Rc);
    }
  }
  return count;
}
/**
 * @brief projection calculation
 *        calculate a row per iteration
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param result
 * @return int the result count.
 */
int proalgo_rowwise(int condition, const idx &size_R,
                    row_store_max *row_max,
                    struct fixed_arrays &result)
{
  int count = 0;
  idx i;
  idx result_size = size_R;
  for (i = 0; i != result_size; ++i)
  {
    if (row_max[i].Ra <= condition && row_max[i].Rc <= condition)
    {
      result.pos_value1[count] = row_max[i].Ra; // value 1
      result.value2[count] = row_max[i].Rc;
      count++;
    }
  }
  return count;
}

int proalgo_cwm_em_simd(int condition, const idx &size_R,
  const int *Ra,
  const int *Rc,
  std::vector<std::pair<int, int>> &result,
  int pre_size)
{

//std::cout << "start simd" << std::endl;

idx read_idx = 0, write_idx = pre_size;
__m512i v_cond = _mm512_set1_epi32(condition);

// 第一次扫描: 处理Ra
for (; read_idx + 16 <= size_R; read_idx += 16) {
__m512i v_ra = _mm512_loadu_si512((__m512i*)(Ra + read_idx));

__mmask16 mask = _mm512_cmple_epi32_mask(v_ra, v_cond);
while (mask) {
//  std::cout << mask << std::endl;
//  std::cout << std::bitset<16>(mask).to_string() << std::endl;
int idx = _tzcnt_u32(mask);
result[write_idx].first = read_idx + idx;      // pos
result[write_idx].second = Ra[read_idx + idx]; // value1
++write_idx;
//std::cout << "before kand" << std::endl;
//std::cout << __tzcnt_u16(mask)<< std::endl;
mask = mask & (mask-1);
//std::cout << "after kand" << std::endl;
}
}
// 处理剩余的元素
for (; read_idx < size_R; ++read_idx) {
if (Ra[read_idx] <= condition) {
result[write_idx].first = read_idx;      // pos
result[write_idx].second = Ra[read_idx]; // value1
++write_idx;
}
}
idx cur_size = write_idx - pre_size;
idx final_write_idx = pre_size;

// 第二次扫描: 处理Rc
for (read_idx = 0; read_idx + 16 <= cur_size; read_idx += 16) {
__m512i v_pos = _mm512_set_epi32(
result[read_idx+15+pre_size].first, result[read_idx+14+pre_size].first,
result[read_idx+13+pre_size].first, result[read_idx+12+pre_size].first,
result[read_idx+11+pre_size].first, result[read_idx+10+pre_size].first,
result[read_idx+9+pre_size].first, result[read_idx+8+pre_size].first,
result[read_idx+7+pre_size].first, result[read_idx+6+pre_size].first,
result[read_idx+5+pre_size].first, result[read_idx+4+pre_size].first,
result[read_idx+3+pre_size].first, result[read_idx+2+pre_size].first,
result[read_idx+1+pre_size].first, result[read_idx+pre_size].first
);

__m512i v_rc = _mm512_i32gather_epi32(v_pos, Rc, 4);
__mmask16 mask = _mm512_cmple_epi32_mask(v_rc, v_cond);

while (mask) {
int idx = _tzcnt_u32(mask);
auto cur_pos = result[read_idx + idx + pre_size].first;
result[final_write_idx].first = result[cur_pos + pre_size].second;  // value 1
result[final_write_idx].second = Rc[cur_pos];                       // value 2
++final_write_idx;
  mask = mask & (mask-1);
}
}

// 处理剩余的元素
for (; read_idx < cur_size; ++read_idx) {
auto cur_pos = result[read_idx + pre_size].first;
if (Rc[cur_pos] <= condition) {
result[final_write_idx].first = result[cur_pos + pre_size].second;  // value 1
result[final_write_idx].second = Rc[cur_pos];                       // value 2
++final_write_idx;
}
}
//std::cout << "end simd" << std::endl;
return final_write_idx;
}

/**
 * @brief projection calculation
 *        calculate one column in one run with early materialization strategy and dynamic vector result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_em(int condition, const idx &size_R,
                   const int *Ra,
                   const int *Rc,
                   std::vector<std::pair<int, int>> &result,
                   int pre_size)
{
  // idx read_idx, cur_size, write_idx;
  // for (read_idx = 0, write_idx = pre_size; read_idx != size_R; ++read_idx)
  // {
  //   if (Ra[read_idx] <= condition)
  //   {
  //     result[write_idx].first = read_idx;      // pos,
  //     result[write_idx].second = Ra[read_idx]; // value1
  //     ++write_idx;
  //   }
  // }
  // cur_size = write_idx - pre_size;
  // for (read_idx = 0, write_idx = pre_size; read_idx != cur_size; ++read_idx)
  // {
  //   auto cur_pos = result[read_idx + pre_size].first;
  //   if (Rc[cur_pos] <= condition)
  //   {
  //     result[write_idx].first = result[cur_pos + pre_size].second; // value 1
  //     result[write_idx].second = Rc[cur_pos];                      // value 2
  //     ++write_idx;
  //   }
  // }

  // return write_idx;
  return proalgo_cwm_em_simd(condition, size_R, Ra, Rc, result, pre_size);
}
/**
 * @brief projection calculation
 *        calculate one column in one run with early materialization strategy and fixed vector result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_em(int condition, const idx &size_R,
                   const int *Ra,
                   const int *Rc,
                   struct fixed_arrays &result,
                   int pre_size)
{

  idx read_idx, cur_size, write_idx;
  for (read_idx = 0, write_idx = pre_size; read_idx != size_R; ++read_idx)
  {
    if (Ra[read_idx] <= condition)
    {
      result.pos_value1[write_idx] = read_idx; // (pos, value1)
      result.value2[write_idx] = Ra[read_idx];
      // result.emplace_back(read_idx, Ra[read_idx]);
      ++write_idx;
    }
  }
  cur_size = write_idx - pre_size;
  for (read_idx = 0, write_idx = pre_size; read_idx != cur_size; ++read_idx)
  {
    auto cur_pos = result.pos_value1[read_idx + pre_size];
    if (Rc[cur_pos] <= condition)
    {
      result.pos_value1[write_idx] = result.value2[cur_pos + pre_size]; // value 1
      result.value2[write_idx] = Rc[cur_pos];                           // value 2
      ++write_idx;
    }
  }

  return write_idx;
}

int proalgo_cwm_lm_idv_simd(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       std::vector<int> &pos1, std::vector<int> &pos2,
                       std::vector<std::pair<int, int>> &result)
{
    idx i = 0;
    __m512i v_cond = _mm512_set1_epi32(condition);

    // 第一步：过滤Ra
    for (; i + 16 <= size_R; i += 16) {
        __m512i v_ra = _mm512_loadu_si512((__m512i*)(Ra + i));

        __mmask16 mask = _mm512_cmple_epi32_mask(v_ra, v_cond);

        while (mask) {
            int idx = _tzcnt_u32(mask);
            pos1.emplace_back(i + idx);
            mask = mask & (mask-1);
        }
    }

    // 处理Ra剩余元素
    for (; i < size_R; ++i) {
        if (Ra[i] <= condition) {
            pos1.emplace_back(i);
        }
    }

    // 第二步：过滤Rc
    i = 0;
    for (; i + 16 <= size_R; i += 16) {
        __m512i v_rc = _mm512_loadu_si512((__m512i*)(Rc + i));

        __mmask16 mask = _mm512_cmple_epi32_mask(v_rc, v_cond);

        while (mask) {
            int idx = _tzcnt_u32(mask);
            pos2.emplace_back(i + idx);
            mask = mask & (mask-1);
        }
    }

    // 处理Rc剩余元素
    for (; i < size_R; ++i) {
        if (Rc[i] <= condition) {
            pos2.emplace_back(i);
        }
    }

    // 第三步：合并结果（保持原有的合并逻辑，因为这部分不适合SIMD优化）
    idx merge_idx = 0;
    idx j = 0;
    for (i = 0, j = 0; i < pos1.size() && j < pos2.size();) {
        if (pos1[i] == pos2[j]) {
            pos1[merge_idx] = pos1[i];
            ++i;
            ++j;
            ++merge_idx;
        }
        else if (pos1[i] > pos2[j]) {
            ++j;
        }
        else {
            ++i;
        }
    }

    // 构建最终结果
    for (i = 0; i != merge_idx; ++i) {
        auto cur_pos = pos1[i];
        result.emplace_back(Ra[cur_pos], Rc[cur_pos]);
    }

    return result.size();
}

/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and independent dynamic vector intermediate results as well as dynamic vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_idv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       std::vector<int> &pos1, std::vector<int> &pos2,
                       std::vector<std::pair<int, int>> &result)
{

//  idx i, j;
//  for (i = 0; i < size_R; ++i)
//  {
//    if (Ra[i] <= condition)
//    {
//      pos1.emplace_back(i);
//    }
//  }
//
//  for (i = 0; i < size_R; ++i)
//  {
//    if (Rc[i] <= condition)
//    {
//      pos2.emplace_back(i);
//    }
//  }
//
//  idx merge_idx = 0;
//  for (i = 0, j = 0; i < pos1.size() && j < pos2.size();)
//  {
//    if (pos1[i] == pos2[j])
//    {
//      pos1[merge_idx] = pos1[i];
//      ++i;
//      ++j;
//      ++merge_idx;
//    }
//    else if (pos1[i] > pos2[j])
//    {
//      ++j;
//    }
//    else
//    {
//      // if pos1[i] < pos2[j]
//      ++i;
//    }
//  }
//
//  for (i = 0; i != merge_idx; ++i)
//  {
//    auto cur_pos = pos1[i];
//    result.emplace_back(Ra[cur_pos], Rc[cur_pos]);
//  }

  return proalgo_cwm_lm_idv_simd(condition, size_R, Ra, Rc, pos1, pos2, result);
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and independent dynamic vector intermediate results as well as dynamic vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_idv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       std::vector<int> &pos1, std::vector<int> &pos2,
                       struct fixed_arrays &result,
                       int pre_size)
{
  idx i, j;
  for (i = 0; i < size_R; ++i)
  {
    if (Ra[i] <= condition)
    {
      pos1.emplace_back(i);
    }
  }

  for (i = 0; i < size_R; ++i)
  {
    if (Rc[i] <= condition)
    {
      pos2.emplace_back(i);
    }
  }

  idx merge_idx = 0;
  for (i = 0, j = 0; i < pos1.size() && j < pos2.size();)
  {
    if (pos1[i] == pos2[j])
    {
      pos1[merge_idx] = pos1[i];
      ++i;
      ++j;
      ++merge_idx;
    }
    else if (pos1[i] > pos2[j])
    {
      ++j;
    }
    else
    {
      // if pos1[i] < pos2[j]
      ++i;
    }
  }

  for (i = 0; i != merge_idx; ++i)
  {
    auto cur_pos = pos1[i];
    int cur_idx = pre_size + i;
    result.pos_value1[cur_idx] = Ra[cur_pos]; // value 1
    result.value2[cur_idx] = Rc[cur_pos];
  }

  return merge_idx;
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and shared dynamic vector intermediate results as well as dynamic vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_sdv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       std::vector<int> &pos,
                       std::vector<std::pair<int, int>> &result)
{
  idx i, cur_size = 0;
  for (i = 0; i < size_R; ++i)
  {
    if (Ra[i] <= condition)
    {
      pos.emplace_back(i);
    }
  }

  for (i = 0; i < pos.size(); ++i)
  {
    if (Rc[pos[i]] <= condition)
    {
      pos[cur_size++] = pos[i];
    }
  }

  for (i = 0; i != cur_size; ++i)
  {
    auto cur_pos = pos[i];
    result.emplace_back(Ra[cur_pos], Rc[cur_pos]);
  }

  return result.size();
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and shared dynamic vector intermediate results as well as fixed vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_sdv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       std::vector<int> &pos,
                       struct fixed_arrays &result,
                       int pre_size)
{
  idx i, cur_size = 0;
  for (i = 0; i < size_R; ++i)
  {
    if (Ra[i] <= condition)
    {
      pos.emplace_back(i);
    }
  }

  for (i = 0; i < pos.size(); ++i)
  {
    if (Rc[pos[i]] <= condition)
    {
      pos[cur_size++] = pos[i];
    }
  }

  for (i = 0; i != cur_size; ++i)
  {
    auto cur_pos = pos[i];
    int cur_idx = i + pre_size;
    result.pos_value1[i] = Ra[cur_pos]; // value 1
    result.value2[i] = Rc[cur_pos];
  }

  return cur_size;
}
int proalgo_cwm_lm_ifv_simd(int condition, const idx &size_R,
  const int *Ra,
  const int *Rc,
  int *pos1, int *pos2,
  std::vector<std::pair<int, int>> &result)
{
idx pos1_idx = 0, pos2_idx = 0;
idx i = 0;
__m512i v_cond = _mm512_set1_epi32(condition);

// 第一步：过滤Ra
for (; i + 16 <= size_R; i += 16) {
__m512i v_ra = _mm512_loadu_si512((__m512i*)(Ra + i));
__mmask16 mask = _mm512_cmple_epi32_mask(v_ra, v_cond);

while (mask) {
int idx = _tzcnt_u32(mask);
pos1[pos1_idx++] = i + idx;
mask = mask & (mask-1);
}
}

// 处理Ra剩余元素
for (; i < size_R; ++i) {
if (Ra[i] <= condition) {
pos1[pos1_idx++] = i;
}
}

// 第二步：过滤Rc
i = 0;
for (; i + 16 <= size_R; i += 16) {
__m512i v_rc = _mm512_loadu_si512((__m512i*)(Rc + i));
__mmask16 mask = _mm512_cmple_epi32_mask(v_rc, v_cond);

while (mask) {
int idx = _tzcnt_u32(mask);
pos2[pos2_idx++] = i + idx;
mask = mask & (mask-1);
}
}

// 处理Rc剩余元素
for (; i < size_R; ++i) {
if (Rc[i] <= condition) {
pos2[pos2_idx++] = i;
}
}

// 第三步：合并结果（保持原有的合并逻辑）
idx merge_idx = 0;
idx j = 0;
for (i = 0, j = 0; i < pos1_idx && j < pos2_idx;) {
if (pos1[i] == pos2[j]) {
pos1[merge_idx] = pos1[i];
++i;
++j;
++merge_idx;
}
else if (pos1[i] > pos2[j]) {
++j;
}
else {
++i;
}
}

// 构建最终结果
result.clear();
result.reserve(merge_idx);  // 预分配空间
for (i = 0; i != merge_idx; ++i) {
auto cur_pos = pos1[i];
result.emplace_back(Ra[cur_pos], Rc[cur_pos]);
}

return result.size();
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and independent fixed vector intermediate results as well as dynamic vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_ifv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       int *pos1, int *pos2,
                       std::vector<std::pair<int, int>> &result)
{
  // idx pos1_idx, pos2_idx, i, j;
  // for (i = 0, pos1_idx = 0; i < size_R; ++i)
  // {
  //   if (Ra[i] <= condition)
  //   {
  //     pos1[pos1_idx] = i;
  //     ++pos1_idx;
  //   }
  // }

  // for (i = 0, pos2_idx = 0; i < size_R; ++i)
  // {
  //   if (Rc[i] <= condition)
  //   {
  //     pos2[pos2_idx] = i;
  //     ++pos2_idx;
  //   }
  // }

  // idx merge_idx = 0;
  // for (i = 0, j = 0; i < pos1_idx && j < pos2_idx;)
  // {
  //   if (pos1[i] == pos2[j])
  //   {
  //     pos1[merge_idx] = pos1[i];
  //     ++i;
  //     ++j;
  //     ++merge_idx;
  //   }
  //   else if (pos1[i] > pos2[j])
  //   {
  //     ++j;
  //   }
  //   else
  //   {
  //     // if pos1[i] < pos2[j]
  //     ++i;
  //   }
  // }

  // for (i = 0; i != merge_idx; ++i)
  // {
  //   auto cur_pos = pos1[i];
  //   result.emplace_back(Ra[cur_pos], Rc[cur_pos]);
  // }

  return proalgo_cwm_lm_ifv_simd(condition, size_R, Ra, Rc, pos1, pos2, result);
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and independent fixed vector intermediate results as well as fixed vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_ifv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       int *pos1, int *pos2,
                       struct fixed_arrays &result,
                       int pre_size)
{
  idx pos1_idx, pos2_idx, i, j;
  for (i = 0, pos1_idx = 0; i < size_R; ++i)
  {
    if (Ra[i] <= condition)
    {
      pos1[pos1_idx] = i;
      ++pos1_idx;
    }
  }

  for (i = 0, pos2_idx = 0; i < size_R; ++i)
  {
    if (Rc[i] <= condition)
    {
      pos2[pos2_idx] = i;
      ++pos2_idx;
    }
  }

  idx merge_idx = 0;
  for (i = 0, j = 0; i < pos1_idx && j < pos2_idx;)
  {
    if (pos1[i] == pos2[j])
    {
      pos1[merge_idx] = pos1[i];
      ++i;
      ++j;
      ++merge_idx;
    }
    else if (pos1[i] > pos2[j])
    {
      ++j;
    }
    else
    {
      // if pos1[i] < pos2[j]
      ++i;
    }
  }

  for (i = 0; i != merge_idx; ++i)
  {
    auto cur_pos = pos1[i];
    int cur_idx = i + pre_size;
    result.pos_value1[cur_idx] = Ra[cur_pos]; // value 1
    result.value2[cur_idx] = Rc[cur_pos];
  }

  return merge_idx;
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and shared fixed vector intermediate results as well as dynamic vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_sfv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       int *pos,
                       std::vector<std::pair<int, int>> &result)
{
  idx pos1_idx, pos2_idx, i, j;
  for (i = 0, pos1_idx = 0; i < size_R; ++i)
  {
    if (Ra[i] <= condition)
    {
      pos[pos1_idx] = i;
      ++pos1_idx;
    }
  }

  for (i = 0, pos2_idx = 0; i < pos1_idx; ++i)
  {
    if (Rc[pos[i]] <= condition)
    {
      pos[pos2_idx] = pos[i];
      ++pos2_idx;
    }
  }

  for (i = 0; i != pos2_idx; ++i)
  {
    auto cur_pos = pos[i];
    result.emplace_back(Ra[cur_pos], Rc[cur_pos]);
  }

  return result.size();
}

/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and shared fixed vector intermediate results as well as fixed vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param pos1
 * @param pos2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_sfv(int condition, const idx &size_R,
                       const int *Ra,
                       const int *Rc,
                       int *pos,
                       struct fixed_arrays &result,
                       int pre_size)
{
  idx pos1_idx, pos2_idx, i, j;
  for (i = 0, pos1_idx = 0; i < size_R; ++i)
  {
    if (Ra[i] <= condition)
    {
      pos[pos1_idx] = i;
      ++pos1_idx;
    }
  }

  for (i = 0, pos2_idx = 0; i < pos1_idx; ++i)
  {
    if (Rc[pos[i]] <= condition)
    {
      pos[pos2_idx] = pos[i];
      ++pos2_idx;
    }
  }

  for (i = 0; i != pos2_idx; ++i)
  {
    auto cur_pos = pos[i];
    int cur_idx = i + pre_size;
    result.pos_value1[cur_idx] = Ra[cur_pos]; // value 1
    result.value2[cur_idx] = Rc[cur_pos];
  }

  return pos2_idx;
}
int proalgo_cwm_lm_ibmp_simd(int condition, const idx &size_R,
  const int *Ra,
  const int *Rc,
  bool *bitmap1, bool *bitmap2,
  std::vector<std::pair<int, int>> &result)
{
idx i = 0;
__m512i v_cond = _mm512_set1_epi32(condition);

// 处理Ra，生成bitmap1
for (; i + 16 <= size_R; i += 16) {
__m512i v_ra = _mm512_loadu_si512((__m512i*)(Ra + i));
__mmask16 mask = _mm512_cmple_epi32_mask(v_ra, v_cond);

// 将16位掩码转换为16个bool值
__m128i expanded_mask = _mm_maskz_set1_epi8(mask, 1);
_mm_storeu_si128((__m128i*)(bitmap1 + i), expanded_mask);
}

// 处理剩余元素
for (; i < size_R; ++i) {
bitmap1[i] = (Ra[i] <= condition);
}

// 处理Rc，生成bitmap2
i = 0;
for (; i + 16 <= size_R; i += 16) {
__m512i v_rc = _mm512_loadu_si512((__m512i*)(Rc + i));
__mmask16 mask = _mm512_cmple_epi32_mask(v_rc, v_cond);

__m128i expanded_mask = _mm_maskz_set1_epi8(mask, 1);
_mm_storeu_si128((__m128i*)(bitmap2 + i), expanded_mask);
}

// 处理剩余元素
for (; i < size_R; ++i) {
bitmap2[i] = (Rc[i] <= condition);
}

// 合并两个bitmap
i = 0;
for (; i + 16 <= size_R; i += 16) {
__m128i v_bitmap1 = _mm_loadu_si128((__m128i*)(bitmap1 + i));
__m128i v_bitmap2 = _mm_loadu_si128((__m128i*)(bitmap2 + i));
__m128i v_result = _mm_and_si128(v_bitmap1, v_bitmap2);
_mm_storeu_si128((__m128i*)(bitmap1 + i), v_result);
}

// 处理剩余元素
for (; i < size_R; ++i) {
bitmap1[i] = bitmap1[i] & bitmap2[i];
}

// 收集结果
result.clear();
result.reserve(size_R); // 预分配空间避免频繁重分配

i = 0;
for (; i + 16 <= size_R; i += 16) {
__m128i v_bitmap = _mm_loadu_si128((__m128i*)(bitmap1 + i));
uint16_t mask = _mm_movemask_epi8(v_bitmap);

while (mask) {
int idx = _tzcnt_u32(mask);
result.emplace_back(Ra[i + idx], Rc[i + idx]);
mask = mask & ~(1 << idx);
}
}

// 处理剩余元素
for (; i < size_R; ++i) {
if (bitmap1[i]) {
result.emplace_back(Ra[i], Rc[i]);
}
}

return result.size();
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and independent bitmap intermediate results as well as dynamic vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param bitmap1
 * @param bitmap2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_ibmp(int condition, const idx &size_R,
                        const int *Ra,
                        const int *Rc,
                        bool *bitmap1, bool *bitmap2,
                        std::vector<std::pair<int, int>> &result)
{
  // idx i;
  // for (i = 0; i != size_R; ++i)
  // {
  //   bitmap1[i] = (Ra[i] <= condition);
  // }

  // for (i = 0; i != size_R; ++i)
  // {
  //   bitmap2[i] = (Rc[i] <= condition);
  // }

  // for (i = 0; i != size_R; ++i)
  // {
  //   bitmap1[i] = (bitmap1[i] & bitmap2[i]);
  // }

  // for (i = 0; i != size_R; ++i)
  // {
  //   if (bitmap1[i])
  //   {
  //     result.emplace_back(Ra[i], Rc[i]);
  //   }
  // }

  return proalgo_cwm_lm_ibmp_simd(condition,size_R,Ra,Rc,bitmap1,bitmap2,result);
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and independent bitmap intermediate results as well as fixed vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param bitmap1
 * @param bitmap2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_ibmp(int condition, const idx &size_R,
                        const int *Ra,
                        const int *Rc,
                        bool *bitmap1, bool *bitmap2,
                        struct fixed_arrays &result,
                        int pre_size)
{
  idx i, cur_size = pre_size;
  for (i = 0; i != size_R; ++i)
  {
    bitmap1[i] = (Ra[i] <= condition);
  }

  for (i = 0; i != size_R; ++i)
  {
    bitmap2[i] = (Rc[i] <= condition);
  }

  for (i = 0; i != size_R; ++i)
  {
    bitmap1[i] = (bitmap1[i] & bitmap2[i]);
  }

  for (i = 0; i != size_R; ++i)
  {
    if (bitmap1[i])
    {
      result.pos_value1[cur_size] = Ra[i]; // value 1
      result.value2[cur_size] = Rc[i];
      cur_size++;
    }
  }

  return cur_size;
}
/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and shared bitmap intermediate results as well as dynamic vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param bitmap1
 * @param bitmap2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_sbmp(int condition, const idx &size_R,
                        const int *Ra,
                        const int *Rc,
                        bool *bitmap,
                        std::vector<std::pair<int, int>> &result)
{
  idx i;
  for (i = 0; i != size_R; ++i)
  {
    bitmap[i] = (Ra[i] <= condition);
  }

  for (i = 0; i != size_R; ++i)
  {
    bitmap[i] = (bitmap[i]) && (Rc[i] <= condition);
  }

  for (i = 0; i != size_R; ++i)
  {
    if (bitmap[i])
    {
      result.emplace_back(Ra[i], Rc[i]);
    }
  }

  return result.size();
}

/**
 * @brief projection calculation
 *        calculate one column in one run with late materialization strategy and shared bitmap intermediate results as well as fixed vector final result
 *
 * @param condition
 * @param size_R
 * @param Ra
 * @param Rc
 * @param bitmap1
 * @param bitmap2
 * @param result
 * @return int the result count.
 */
int proalgo_cwm_lm_sbmp(int condition, const idx &size_R,
                        const int *Ra,
                        const int *Rc,
                        bool *bitmap,
                        struct fixed_arrays &result,
                        int pre_size)
{
  idx i, cur_size = pre_size;
  for (i = 0; i != size_R; ++i)
  {
    bitmap[i] = (Ra[i] <= condition);
  }

  for (i = 0; i != size_R; ++i)
  {
    bitmap[i] = (bitmap[i]) && (Rc[i] <= condition);
  }

  for (i = 0; i != size_R; ++i)
  {
    if (bitmap[i])
    {
      result.pos_value1[cur_size] = Ra[i]; // value 1
      result.value2[cur_size] = Rc[i];
      cur_size++;
    }
  }

  return cur_size;
}
/**
 * @brief projection calculation by Column-wise query processing model with early materialization strategy
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_cwm_em(const idx &size_R,
                         const T *Ra, const T *Rc,
                         const std::vector<idx> &conditions,
                         std::ofstream &proalgo_timefile)
{
  /*dynamic vector result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with early materialization strategy and dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_em(conditions[select_idx], size_R, Ra, Rc, result, count);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "early materialization strategy"
                     << "\t"
                     << ""
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_cwm_em_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*fixed vector result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with early materialization strategy and fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_em(conditions[select_idx], size_R, Ra, Rc, result, count);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "early materialization strategy"
                     << "\t"
                     << ""
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_cwm_em_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Vector-wise query processing model with early materialization strategy
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_vwm_em(const idx &size_R,
                         const T *Ra, const T *Rc,
                         const std::vector<idx> &conditions,
                         const std::vector<idx> &conditions_lsr,
                         std::ofstream &proalgo_timefile,
                         std::ofstream &proalgo_lsr_timefile)
{
  /*dynamic vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with early materialization strategy and dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_em(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, result, count);
    }

    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "early materialization strategy"
                     << "\t"
                     << ""
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_vwm_em_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with early materialization strategy and fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_em(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "early materialization strategy"
                     << "\t"
                     << ""
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_em_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with early materialization strategy and fixed vector final result in low selection rate test" << std::endl;

  for (idx select_idx = 0; select_idx != conditions_lsr.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions_lsr[select_idx] / 10 << " %, total selection rate " << pow((double)conditions_lsr[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_em(conditions_lsr[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << (double)pow(conditions_lsr[select_idx] / 10, 2) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_lsr_timefile << "Vector-wise query processing"
                     << "\t"
                     << "early materialization strategy"
                     << "\t"
                     << ""
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "Vector-wise query processing with early materialization strategy and fixed vector final result"
                     << "\t"
                     << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Column-wise query processing model with late materialization strategy and dynamic vector Intermediate results
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_cwm_lm_dv(const idx &size_R,
                            const T *Ra, const T *Rc,
                            const std::vector<idx> &conditions,
                            std::ofstream &proalgo_timefile)
{
  /*independent dynamic vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and independent dynamic vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    std::vector<int> pos1, pos2;
    pos1.reserve(size_R);
    pos2.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_idv(conditions[select_idx], size_R, Ra, Rc, pos1, pos2, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent dynamic vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_idvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*independent dynamic vector intermediate results as well as fixed vector result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and independent dynamic vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];

    std::vector<int> pos1, pos2;
    pos1.reserve(size_R);
    pos2.reserve(size_R);

    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_idv(conditions[select_idx], size_R, Ra, Rc, pos1, pos2, result, count);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent dynamic vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_idvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared dynamic vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<int> pos;
    std::vector<std::pair<int, int>> result;
    pos.reserve(size_R);
    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_sdv(conditions[select_idx], size_R, Ra, Rc, pos, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared dynamic vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_sdvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*shared dynamic vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<int> pos;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    pos.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_sdv(conditions[select_idx], size_R, Ra, Rc, pos, result, count);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared dynamic vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_sdvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Vector-wise query processing model with late materialization strategy and dynamic vector Intermediate results
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_vwm_lm_dv(const idx &size_R,
                            const T *Ra, const T *Rc,
                            const std::vector<idx> &conditions,
                            std::ofstream &proalgo_timefile)
{
  /*independent dynamic vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and independent dynamic vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    std::vector<int> pos1, pos2;
    pos1.reserve(size_v);
    pos2.reserve(size_v);
    idx vec_num = DATA_NUM / size_v;
    timeval start, end;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_idv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos1, pos2, result);
      pos1.clear();
      pos2.clear();
    }

    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent dynamic vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_idvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*independent dynamic vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and independent dynamic vector intermediate results and fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];

    std::vector<int> pos1, pos2;
    pos1.reserve(size_v);
    pos2.reserve(size_v);

    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += proalgo_cwm_lm_idv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos1, pos2, result, count);
      pos1.clear();
      pos2.clear();
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent dynamic vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_idvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared dynamic vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<int> pos;
    std::vector<std::pair<int, int>> result;
    pos.reserve(size_v);
    result.reserve(size_R);
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_sdv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos, result);
      pos.clear();
    }

    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared dynamic vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_sdvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared dynamic vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<int> pos;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    pos.reserve(size_v);
    idx vec_num = DATA_NUM / size_v;
    timeval start, end;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += proalgo_cwm_lm_sdv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos, result, count);
      pos.clear();
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared dynamic vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_sdvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Column-wise query processing model with late materialization strategy and fixed vector intermediate results as well as dynamic vector final result
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_cwm_lm_fv(const idx &size_R,
                            const T *Ra, const T *Rc,
                            const std::vector<idx> &conditions,
                            std::ofstream &proalgo_timefile)
{
  /*independent fixed vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and independent fixed vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos1 = new int[size_R];
    int *pos2 = new int[size_R];
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_ifv(conditions[select_idx], size_R, Ra, Rc, pos1, pos2, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] pos1;
    delete[] pos2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent fixed vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_ifvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*independent fixed vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and independent fixed vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos1 = new int[size_R];
    int *pos2 = new int[size_R];
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_ifv(conditions[select_idx], size_R, Ra, Rc, pos1, pos2, result, count);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] pos1;
    delete[] pos2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent fixed vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_ifvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared fixed vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and shared fixed vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos = new int[size_R];
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_sfv(conditions[select_idx], size_R, Ra, Rc, pos, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] pos;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared fixed vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_sfvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared fixed vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and shared fixed vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos = new int[size_R];
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_sfv(conditions[select_idx], size_R, Ra, Rc, pos, result, count);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] pos;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared fixed vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_sfvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Vector-wise query processing model with late materialization strategy and fixed vector intermediate results as well as dynamic vector final result
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_vwm_lm_fv(const idx &size_R,
                            const T *Ra, const T *Rc,
                            const std::vector<idx> &conditions,
                            const std::vector<idx> &conditions_lsr,
                            std::ofstream &proalgo_timefile,
                            std::ofstream &proalgo_lsr_timefile)
{
  /*independent fixed vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and independent fixed vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos1 = new int[size_v];
    int *pos2 = new int[size_v];
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_ifv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos1, pos2, result);
    }

    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] pos1;
    delete[] pos2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent fixed vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_ifvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*independent fixed vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and independent fixed vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos1 = new int[size_v];
    int *pos2 = new int[size_v];
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += proalgo_cwm_lm_ifv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos1, pos2, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] pos1;
    delete[] pos2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent fixed vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_ifvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared fixed vector intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and shared fixed vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos = new int[size_v];
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    idx vec_num = DATA_NUM / size_v;
    timeval start, end;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_sfv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos, result);
    }

    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] pos;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared fixed vector intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_sfvi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared fixed vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and shared fixed vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos = new int[size_v];
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += proalgo_cwm_lm_sfv(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] pos;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared fixed vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_sfvi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*shared fixed vector intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and shared fixed vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions_lsr.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions_lsr[select_idx] / 10 << " %, total selection rate " << pow((double)conditions_lsr[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    int *pos = new int[size_v];
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count += proalgo_cwm_lm_sfv(conditions_lsr[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, pos, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] pos;
    std::cout << "          Result count of selection rate " << (double)pow(conditions_lsr[select_idx] / 10, 2) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_lsr_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared fixed vector intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_sfvi_fvf"
                     << "\t"
                     << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Column-wise query processing model with late materialization strategy and bitmap
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_cwm_lm_bmp(const idx &size_R,
                             const T *Ra, const T *Rc,
                             const std::vector<idx> &conditions,
                             std::ofstream &proalgo_timefile)
{
  /*independent bitmap intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and independent bitmap intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    bool *bitmap1 = new bool[size_R];
    bool *bitmap2 = new bool[size_R];
    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_ibmp(conditions[select_idx], size_R, Ra, Rc, bitmap1, bitmap2, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] bitmap1;
    delete[] bitmap2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent bitmap intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_ibmpi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*independent bitmap intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and independent bitmap intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    bool *bitmap1 = new bool[size_R];
    bool *bitmap2 = new bool[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_ibmp(conditions[select_idx], size_R, Ra, Rc, bitmap1, bitmap2, result, count);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] bitmap1;
    delete[] bitmap2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent bitmap intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_ibmpi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }

  /*shared bitmap intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    bool *bitmap = new bool[size_R];

    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_sbmp(conditions[select_idx], size_R, Ra, Rc, bitmap, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] bitmap;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared bitmap intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_sbmpi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*shared bitmap intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Column-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    bool *bitmap = new bool[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_cwm_lm_sbmp(conditions[select_idx], size_R, Ra, Rc, bitmap, result, count);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] bitmap;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Column-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared bitmap intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_cwm_lm_sbmpi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Vector-wise query processing model with late materialization strategy and bitmap
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_vwm_lm_bmp(const idx &size_R,
                             const T *Ra, const T *Rc,
                             const std::vector<idx> &conditions,
                             const std::vector<idx> &conditions_lsr,
                             std::ofstream &proalgo_timefile,
                             std::ofstream &proalgo_lsr_timefile)
{
  /*independent bitmap intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and independent bitmap intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    bool *bitmap1 = new bool[size_v];
    bool *bitmap2 = new bool[size_v];
    result.reserve(size_R);
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_ibmp(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, bitmap1, bitmap2, result);
    }

    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] bitmap1;
    delete[] bitmap2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent bitmap intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_ibmpi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*independent bitmap intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and independent bitmap intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    bool *bitmap1 = new bool[size_v];
    bool *bitmap2 = new bool[size_v];
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_ibmp(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, bitmap1, bitmap2, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] bitmap1;
    delete[] bitmap2;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent bitmap intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_ibmpi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*independent bitmap intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and independent bitmap intermediate results as well as fixed vector final result in low selection rate test" << std::endl;

  for (idx select_idx = 0; select_idx != conditions_lsr.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions_lsr[select_idx] / 10 << " %, total selection rate " << pow((double)conditions_lsr[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    bool *bitmap1 = new bool[size_v];
    bool *bitmap2 = new bool[size_v];
    timeval start, end;
    idx vec_num = DATA_NUM / size_v;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_ibmp(conditions_lsr[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, bitmap1, bitmap2, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] bitmap1;
    delete[] bitmap2;
    std::cout << "          Result count of selection rate " << (double)pow(conditions_lsr[select_idx] / 10, 2) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_lsr_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "independent bitmap intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_ibmpi_fvf"
                     << "\t"
                     << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                     << "\t"
                     << ms << std::endl;
  }
  /*shared bitmap intermediate results as well as dynamic vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    bool *bitmap = new bool[size_v];
    idx vec_num = DATA_NUM / size_v;
    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_sbmp(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, bitmap, result);
    }

    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    delete[] bitmap;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared bitmap intermediate results"
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_sbmpi_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  /*shared bitmap intermediate results as well as fixed vector final result*/
  std::cout << ">>> start projection calculation by Vector-wise query processing model with late materialization strategy and shared dynamic vector intermediate results as well as fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    bool *bitmap = new bool[size_v];
    idx vec_num = DATA_NUM / size_v;
    timeval start, end;
    gettimeofday(&start, NULL);
    for (idx i = 0; i != vec_num; ++i)
    {
      count = proalgo_cwm_lm_sbmp(conditions[select_idx], size_v, Ra + i * size_v, Rc + i * size_v, bitmap, result, count);
    }

    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    delete[] bitmap;
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Vector-wise query processing"
                     << "\t"
                     << "late materialization strategy"
                     << "\t"
                     << "shared bitmap intermediate results"
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_vwm_lm_sbmpi_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Column-wise query processing model with late materialization strategy
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_cwm_lm(const idx &size_R,
                         const T *Ra, const T *Rc,
                         const std::vector<idx> &conditions,
                         std::ofstream &proalgo_timefile)
{
  /*1. dynamic vector*/
  test_proalgo_cwm_lm_dv(DATA_NUM, Ra, Rc, conditions, proalgo_timefile);
  /*2. fixed vector*/
  test_proalgo_cwm_lm_fv(DATA_NUM, Ra, Rc, conditions, proalgo_timefile);
  /*2. bitmap*/
  test_proalgo_cwm_lm_bmp(DATA_NUM, Ra, Rc, conditions, proalgo_timefile);
}
/**
 * @brief projection calculation by Vector-wise query processing model with late materialization strategy
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_vwm_lm(const idx &size_R,
                         const T *Ra, const T *Rc,
                         const std::vector<idx> &conditions,
                         const std::vector<idx> &conditions_lsr,
                         std::ofstream &proalgo_timefile,
                         std::ofstream &proalgo_lsr_timefile)
{
  /*1. dynamic vector*/
  test_proalgo_vwm_lm_dv(DATA_NUM, Ra, Rc, conditions, proalgo_timefile);
  /*2. fixed vector*/
  test_proalgo_vwm_lm_fv(DATA_NUM, Ra, Rc, conditions, conditions_lsr, proalgo_timefile, proalgo_lsr_timefile);
  /*2. bitmap*/
  test_proalgo_vwm_lm_bmp(DATA_NUM, Ra, Rc, conditions, conditions_lsr, proalgo_timefile, proalgo_lsr_timefile);
}
/**
 * @brief projection calculation by Row-wise query processing model
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_rowwise_model(const idx &size_R,
                                row_store_min *row_min,
                                row_store_max *row_max,
                                const std::vector<idx> &conditions,
                                const std::vector<idx> &conditions_lsr,
                                std::ofstream &proalgo_timefile,
                                std::ofstream &proalgo_lsr_timefile)
{
  std::cout << ">>> start projection calculation by Row-wise query processing model in cache with dynamic vector final result" << std::endl;
  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    timeval start, end;
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    gettimeofday(&start, NULL);
    count = proalgo_rowwise(conditions[select_idx], size_R, row_min, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Row-wise query processing model in cache"
                     << "\t"
                     << ""
                     << "\t"
                     << ""
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_rowwise_rwmic_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  std::cout << ">>> start projection calculation by Row-wise query processing model in cache with fixed vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_rowwise(conditions[select_idx], size_R, row_min, result);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Row-wise query processing model in cache"
                     << "\t"
                     << ""
                     << "\t"
                     << ""
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_rowwise_rwmic_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  std::cout << ">>> start projection calculation by Row-wise query processing model in cache with fixed vector final result in low selection rate test" << std::endl;

  for (idx select_idx = 0; select_idx != conditions_lsr.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions_lsr[select_idx] / 10 << " %, total selection rate " << pow((double)conditions_lsr[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_rowwise(conditions_lsr[select_idx], size_R, row_min, result);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << (double)pow(conditions_lsr[select_idx] / 10, 2) / pow(100, 2) << "% is " << count << "/" << DATA_NUM << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_lsr_timefile << "Row-wise query processing model in cache"
                         << "\t"
                         << ""
                         << "\t"
                         << ""
                         << "\t"
                         << "fixed vector final result"
                         << "\t"
                         << "proalgo_rowwise_rwmic_fvf"
                         << "\t"
                         << log10(LSR_BASE * pow(LSR_STRIDE, select_idx))
                         << "\t"
                         << ms << std::endl;
  }
  std::cout << ">>> start projection calculation by Row-wise query processing model out of cache with dynamic vector final result" << std::endl;

  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    std::vector<std::pair<int, int>> result;
    result.reserve(size_R);
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_rowwise(conditions[select_idx], size_R, row_max, result);
    gettimeofday(&end, NULL);
    result.clear();
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Row-wise query processing model out of cache"
                     << "\t"
                     << ""
                     << "\t"
                     << ""
                     << "\t"
                     << "dynamic vector final result"
                     << "\t"
                     << "proalgo_rowwise_rwmoc_dvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
  std::cout << ">>> start projection calculation by Row-wise query processing model out of cache with fixed vector final result" << std::endl;
  for (idx select_idx = 0; select_idx != conditions.size(); ++select_idx)
  {
    std::cout << "      column selection rate " << (double)conditions[select_idx] / 10 << "%, total selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "%" << std::endl;
    int count = 0;
    struct fixed_arrays result;
    result.pos_value1 = new idx[size_R];
    result.value2 = new idx[size_R];
    timeval start, end;
    gettimeofday(&start, NULL);
    count = proalgo_rowwise(conditions[select_idx], size_R, row_max, result);
    gettimeofday(&end, NULL);
    double ms = calc_ms(end, start);
    std::cout << "          Result count of selection rate " << pow((double)conditions[select_idx] / 10, 2) / pow(100, 2) * 100 << "% is " << count << "/" << size_R << std::endl;
    std::cout << "          Time: " << ms << "ms" << std::endl;
    proalgo_timefile << "Row-wise query processing model out of cache"
                     << "\t"
                     << ""
                     << "\t"
                     << ""
                     << "\t"
                     << "fixed vector final result"
                     << "\t"
                     << "proalgo_rowwise_rwmoc_fvf"
                     << "\t"
                     << 0.1 * (select_idx + 1)
                     << "\t"
                     << ms << std::endl;
  }
}
/**
 * @brief projection calculation by Column-wise query processing model
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_columnwise_model(const idx &size_R,
                                   const T *Ra, const T *Rc,
                                   const std::vector<idx> &conditions,
                                   std::ofstream &proalgo_timefile)
{
  /*1. early materialization strategy*/
  test_proalgo_cwm_em(DATA_NUM, Ra, Rc, conditions, proalgo_timefile);
  /*1. late materialization strategy*/
  test_proalgo_cwm_lm(DATA_NUM, Ra, Rc, conditions, proalgo_timefile);
}
/**
 * @brief projection calculation by Vector-wise query processing model
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo_vectorwise_model(const idx &size_R,
                                   const T *Ra, const T *Rc,
                                   const std::vector<idx> &conditions,
                                   const std::vector<idx> &conditions_lsr,
                                   std::ofstream &proalgo_timefile,
                                   std::ofstream &proalgo_lsr_timefile)
{
  /*1. early materialization strategy*/
  test_proalgo_vwm_em(DATA_NUM, Ra, Rc, conditions, conditions_lsr, proalgo_timefile, proalgo_lsr_timefile);
  /*1. late materialization strategy*/
  test_proalgo_vwm_lm(DATA_NUM, Ra, Rc, conditions, conditions_lsr, proalgo_timefile, proalgo_lsr_timefile);
}
/**
 * @brief projection algorithm implementation and testing
 * @param size_R
 * @param Ra
 * @param Rc
 * @param conditions
 * @param proalgo_timefile
 * @return void
 */
void test_proalgo(const idx &size_R,
                  const T *Ra, const T *Rc,
                  const std::vector<idx> &conditions,
                  const std::vector<idx> &conditions_lsr,
                  row_store_min *row_min,
                  row_store_max *row_max,
                  std::ofstream &proalgo_timefile,
                  std::ofstream &proalgo_lsr_timefile)
{
  /*1.Row-wise query processing model*/
  test_proalgo_rowwise_model(DATA_NUM, row_min, row_max, conditions, conditions_lsr, proalgo_timefile, proalgo_lsr_timefile);
  /*2.Column-wise query processing model*/
  test_proalgo_columnwise_model(DATA_NUM, Ra, Rc, conditions, proalgo_timefile);
  /*3.Vector-wise query processing model*/
  test_proalgo_vectorwise_model(DATA_NUM, Ra, Rc, conditions, conditions_lsr, proalgo_timefile, proalgo_lsr_timefile);
}
int main()
{

  std::ofstream proalgo_timefile, proalgo_lsr_timefile;

  proalgo_lsr_timefile.open(PROALGO_LSR_TIME_FILE, std::ios::out | std::ios::trunc);
  proalgo_lsr_timefile << "Query Processing Model"
                       << "\t"
                       << "Materialization strategy"
                       << "\t"
                       << "Intermediate Result Type"
                       << "\t"
                       << "Final Result Type"
                       << "\t"
                       << "Query Processing Model with different Materialization strategy、Intermediate Result Type and Final Result Type"
                       << "\t"
                       << "Lg(Selection Rate)"
                       << "\t"
                       << "Runtimes(ms)" << std::endl;

  proalgo_timefile.open(PROALGO_TIME_FILE, std::ios::out | std::ios::trunc);
  proalgo_timefile << "Query Processing Model"
                   << "\t"
                   << "Materialization strategy"
                   << "\t"
                   << "Intermediate Result Type"
                   << "\t"
                   << "Final Result Type"
                   << "\t"
                   << "Query Processing Model with different Materialization strategy、Intermediate Result Type and Final Result Type"
                   << "\t"
                   << "Selection Rate"
                   << "\t"
                   << "Runtimes(ms)" << std::endl;

  T *Ra = new T[DATA_NUM];
  T *Rc = new T[DATA_NUM];
  std::vector<int> conditions;
  std::vector<int> conditions_lsr;
  gen_data(DATA_NUM, Ra, Rc, row_min, row_max);
  gen_conditions(conditions, conditions_lsr);
  /*Projection algorithm based on different query models*/
  test_proalgo(DATA_NUM, Ra, Rc, conditions, conditions_lsr, row_min, row_max, proalgo_timefile, proalgo_lsr_timefile);
  delete[] Ra;
  delete[] Rc;
  conditions.clear();
  proalgo_timefile.close();
}