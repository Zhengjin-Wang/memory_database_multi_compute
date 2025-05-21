#ifndef Metadata_H
#define Metadata_H
#include <string>
typedef int idx;
typedef int T;
#define STACKSIZE 8388608
idx DATA_NUM = 1 << 26;    // the version of the src
// idx DATA_NUM = 1024;
// idx DATA_NUM = 64;
const int DATA_MAX = 1000;     //max for low selection rate test   selet 100;project 1000
// const int DATA_MAX = 1000;     //max for low selection rate test
// const int DATA_MAX_fixed_rate = 100;     //max for low selection rate test
const idx CONST_TEST_NUM = 10;   // 10 tests with const change of selection rate stride
const idx CONST_TEST_NUM_LSR = 6; // 9 tests with change for  low selection rate 
const double LSR_BASE = 0.000001; //low selection rate base selection rate
const int LSR_STRIDE= 10;           //stride of low selection rate
const float CONST_BASE = 0.1;       // const stride base selection rate
const float CONST_STRIDE = 0.1;     // stride of selection rate
const char* SELALGO_PTHREAD_TIMEFILE = "./log/select/selalgo_pthread_timefile.txt";
const char* SELALGO_PTHREAD_LSR_TIMEFILE = "./log/select/selalgo_pthread_lsr_timefile.txt";
const char* SELALGO_MODEL_SIMD_FILE = "./log/select/selalgo_model_simd_test.tsv";    // the result time file for select algorithm implementation and testing
const char* SELALGO_MODEL_LSR_SIMD_FILE = "./log/select/selalgo_model_lsr_simd_test.tsv";    // the result time file for select algorithm implementation and low selection rate testing
const char* SELALGO_TIME_FILE = "./log/select/selalgo_test.tsv"; // the result time file for select algorithm implementation and testing
const char* SELALGO_MODEL_TIME_FILE = "./log/select/selalgo_model_test.tsv"; // the result time file for select algorithm implementation and query model testing
const char* SELALGO_MODEL_LSR_TIME_FILE = "./log/select/selalgo_model_lsr_test.tsv"; // the result time file for select algorithm implementation and low selection rate testing
const char* CASESTUDY_TIME_FILE = "./log/select/casestudy_test.tsv"; // the result time file
const char* CASESTUDY_LSR_TIME_FILE = "./log/select/casestudy_lsr_test.tsv"; // the result time file
const char* PROALGO_TIME_FILE = "./log/project/proalgo_test.tsv";
const char* PROALGO_LSR_TIME_FILE = "./log/project/proalgo_lsr_test.tsv";
const char* JOINALGO_TEST1_TIME_FILE = "./log/join/joinalgo_test1.tsv"; //the result time file for join test 1
const char* JOINALGO_TEST2_TIME_FILE = "./log/join/joinalgo_test2.tsv"; //the result time file for join test 2
const char* GROUPALGO_TEST_TIME_FILE = "./log/group/groupalgo_test.tsv"; //the result time file for group test
const char* AGGALGO_TEST_TIME_FILE = "./log/aggregation/aggalgo_test.tsv"; //the result time file for aggregation test
const char* AGGALGO_VEC_TEST_TIME_FILE = "./log/aggregation/aggalgo_vec_test.tsv"; //the result time file for aggregation vector test
const char* STARJOINALGO_TEST_TIME_FILE = "./log/starjoin/starjoin_test.tsv"; //the result time file for starjoin test  
const char* OLAPCORE_TEST_TIME_FILE = "./log/multi_compute_operator/olapcore_test.tsv"; //the result time file for OLAPcore test 
const char* GPUOLAPCORE_TEST_TIME_FILE = "./log/gpu_multi_compute_operator/gpuolapcore_test.tsv"; //the result time file for OLAPcore test 
const char* lineitem_data_dir = "./dbgen/lineitem.csv"; //TPCH lineitem data_dir
const char* partsupp_data_dir = "./dbgen/partsupp.csv"; //TPCH partsupp data_dir
const char* orders_data_dir = "./dbgen/orders.csv"; //TPCH orders data_dir
const char* part_data_dir = "./dbgen/part.csv"; //TPCH part data_dir
const char* supplier_data_dir = "./dbgen/supplier.csv"; //TPCH supplier data_dir
const char* customer_data_dir = "./dbgen/customer.csv"; //TPCH customer data_dir
const char* nation_data_dir = "./dbgen/nation.csv"; //TPCH nation data_dir
const char* region_data_dir = "./dbgen/region.csv"; //TPCH region data_dir
const int M_VALUE = 5;                          // value of M1 and M2
const int DATA_NUM_BASE = 6000000;
const int GROUP_EXP_MIN = 5;                    // group num min 2^5
const int GROUP_EXP_MAX = 26;                   // group num max 2^26
const int L_TAX_MIN = 1;                   // tax value min 0.01
const int L_TAX_MAX = 20;                   // tax value min 0.01
const int L_QUANTITY_MIN = 1;                   // quantity value min 0.01
const int L_QUANTITY_MAX = 10;                   // quantity value min 0.01
const int L_EXPTENDEDPRICE_MIN = 5;                   // extendedprice value min 0.01
const int L_EXPTENDEDPRICE_MAX = 20;                   // extendedprice value min 0.01
const int VEC_SIZE_MAX = 24;                   // vec_size max 2 ^ 24
const char* L1_cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index0/size"; // File path for storing L1cache size
const char* L2_cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index2/size"; // File path for storing L2cache size
const char* L3_cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index3/size"; // File path for storing L3cache size
const idx size_v = 1024;//Vector length
const idx independent_bitmap_size_v = 1024; // once one __mm512i
constexpr int8_t DIM_NULL = INT8_MAX;
constexpr int GROUP_NULL = INT16_MAX;
const size_t LINEORDER_BASE = 6000000;                // lineorder base num
const size_t CUSTOMER_BASE = 30000;                   // customer base num
const size_t SUPPLIER_BASE = 2000;                    // supplier base num
const size_t PART_BASE = 200000;                      // part base num
const size_t DATE_BASE = 7 * 365;                     // date base num
const int GROUP_BITS_TABLE = 4;                   // group num on each table
enum Selalgo_Branch {
    BRANCH_ONE_TWO_THREE = 0,
    BRANCH_ONE_TWO,
    BRANCH_ONE,
    NON_BRANCH
};
template <typename T>
struct Select_Data
{
  const T *sel_col1;
  const T *sel_col2;
  int8_t op;        // Filter symbols
  int8_t col2_flag; // Single value or column
  int8_t select_flag; 
  int *pre_bmp;
  int *res_bmp;
};
struct Select_Node
{
  int select_num = 2;
  Select_Data<int> select_data_int[2];
  Select_Data<std::string> select_data_string[2];
  int8_t logic;//&& OR ||
  int8_t col_type[2]; //int or string
  int col_length;
  std::string tablename;
};
template <typename T>
struct pth_st
{
  int tid;
  pthread_barrier_t *barrier;
  const T *sel_col1;
  const T *sel_col2;
  std::string tablename;
  int8_t op;
  int8_t col2_flag;
  int8_t select_flag;
  int *pre_bmp;
  int *res_bmp;
  int8_t logic;
  int startindex;
  int comline; // 维表行数
};
struct fixed_arrays
{
  idx *pos_value1;
  idx *value2;
  idx array_size = 0;
};
struct row_store_min
{
  idx Ra;
  char Rb[52];
  idx Rc;
};
struct row_store_max
{
  idx Ra;
  char Rb[64];
  idx Rc;
};
struct relation_t
{
  int32_t *key;
  int32_t *payload;
  uint64_t num_tuples;
  double payload_rate;
  int table_size;
  
};
struct Dimvec_array_numa
{
  int8_t *dimvec[4];
};
struct Fk_array_numa
{
  int32_t *fk[4];
};
struct create_arg_t {
    relation_t          rel;
    int64_t             firstkey;
    int64_t             maxid;
    uint64_t            ridstart;
    pthread_barrier_t * barrier;
    int tid;
};
enum TABLE_NAME {
    customer, 
    supplier,
    part, 
    date, 
    lineorder
};
 struct param_t
{
    uint32_t nthreads;
    double sf;
    uint32_t d_groups;
    uint32_t s_groups;
    uint32_t p_groups;
    uint32_t c_groups;
    double d_sele;
    double s_sele;
    double p_sele;
    double c_sele;
    int d_bitmap;
    int s_bitmap;
    int p_bitmap;
    int c_bitmap;
    int basic_numa;
    int sqlnum;
};
struct pth_rowolapcoret
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  int8_t **dimvec_array;
  int32_t **fk_array;
  int dimvec_nums;
  const int *orders;
  uint32_t *group_vector;
  int32_t * M1;
  int32_t * M2;
  const int * factor;
};
struct pth_vwmolapcoret
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  int8_t **dimvec_array;
  int32_t **fk_array;
  int dimvec_nums;
  int *orders;
  int64_t * OID;
  int16_t * groupID;
  uint32_t *group_vector;
  int32_t * M1;
  int32_t * M2;
  int * index;
  int * factor;
};
struct pth_vwmolapcoret_numa
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  Dimvec_array_numa *dimvec_array_numa;
  Fk_array_numa *fk_array_numa;
  int dimvec_nums;
  int *orders;
  int64_t * OID;
  int16_t * groupID;
  uint32_t *group_vector;
  int32_t * M1;
  int32_t * M2;
  int * index;
  int * factor;
};
struct pth_cwmjoint
{
  int join_id;
  int64_t start;
  int64_t num_tuples;
  int8_t *dimvec;
  int32_t *fk;
  int64_t * OID;
  int16_t * groupID;
  int *index;
  int factor;
  int tid;
  int32_t *M1;
  int32_t *M2;
};
struct pth_cwmaggt
{
  int64_t start;
  int64_t num_tuples;
  int64_t * OID;
  int16_t * groupID;
  int * index;
  int32_t * M1;
  int32_t * M2;
  uint32_t * group_vector;
};
extern row_store_min row_min[67108864];
extern row_store_max row_max[67108864];

const Selalgo_Branch SELALGO_BRANCH[] = {BRANCH_ONE_TWO_THREE, NON_BRANCH};

const Selalgo_Branch CASE_COMBINED_BRANCH[] = {BRANCH_ONE_TWO_THREE, BRANCH_ONE_TWO, BRANCH_ONE, NON_BRANCH};

const Selalgo_Branch CASE_MULTIPASS_BRANCH[] = {BRANCH_ONE_TWO, BRANCH_ONE};

/* pthread */ 
#define MAX_NODES 512

struct relation_t2{
  // int num_tuples;    // tuples number of this part
  const T* col;
  int start;    // current thread start position
};
struct arg_bm {
    int32_t             tid;
    // hashtable_t *       ht;
    // int8_t *            bitmap;
    int  num_tuples;    // the tuples' number of this part
    int condition;
    relation_t2          Ra;
    relation_t2          Rb;
    relation_t2          Rc;
    relation_t2          Rd;
    pthread_barrier_t * barrier;
    int             num_results;
    #ifndef NO_TIMING
    /* stats about the thread */
    uint64_t timer1, timer2, timer3;
    struct timeval start, end;
    #endif
} ;
struct arg_bm_numa {
    int32_t             tid;
    // hashtable_t *       ht;
    // int8_t *            bitmap;
    int  num_tuples;    // the tuples' number of this part
    int condition;
    int *vec_a_p;
    int *vec_b_p;
    int *vec_c_p;
    int *vec_d_p;
    
    int *f_sv_ra;
    int *f_sv_rb;
    int *f_sv_rc;
    pthread_barrier_t * barrier;
    int             num_results;
    #ifndef NO_TIMING
    /* stats about the thread */
    uint64_t timer1, timer2, timer3;
    struct timeval start, end;
    #endif
} ;
#endif