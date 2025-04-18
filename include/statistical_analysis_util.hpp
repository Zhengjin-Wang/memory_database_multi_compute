#ifndef Statistical_analysis_util_H
#define Statistical analysis_util_H
#include "metadata.h"
#include <ctime>
#include <sys/time.h>

/**
 * @brief calculate ms using timeval struct
 * 
 * @param end end_time
 * @param start start_time
 * @return double end - start
 */
inline double calc_ms(timeval end, timeval start) {
    return (end.tv_sec - start.tv_sec)*1000 + (end.tv_usec - start.tv_usec)/1000.0;
}
/**
 * @brief find the maximum function
 * @return idx maxinum
 */
 /*idx Find_Maxnum()
 {
   idx tmp_max = 0;
   for(int i = 0; i < test_num; i++)
   {
     for(int j = 0; j < CONST_TEST_NUM; j++)
     {
       if(tmp_max<runtimes_set[i].runtimes[j])
         tmp_max = runtimes_set[i].runtimes[j];
     }
   }
   return tmp_max;
 }*/

#endif