cc = g++
nvcc = nvcc
OBJ_DIR = ./obj
SRC_DIR = ./code
CFLAGS=-I./include -lnuma -g  -mcmodel=large -O0 -std=c++17 -lpthread  -Wno-deprecated -mavx512bw -mavx512vl -mavx512f -mavx512cd -mavx512dq -msse -Wmultichar -mavx512vpopcntdq -mbmi -mbmi2  -larrow -lparquet -larrow_dataset
CUDAFLAGS=-I./include -L/usr/local/lib  -O0 -g  -std=c++11 -lnuma
SRC_SEL = $(wildcard $(SRC_DIR)/select/*.cpp)
OBJ_SEL = $(patsubst $(SRC_DIR)/select/%.cpp, $(OBJ_DIR)/select/%.o, $(SRC_SEL))
SRC_PRO = $(wildcard $(SRC_DIR)/project/*.cpp)
OBJ_PRO = $(patsubst $(SRC_DIR)/project/%.cpp, $(OBJ_DIR)/project/%.o, $(SRC_PRO))
SRC_JOIN = $(wildcard $(SRC_DIR)/join/*.cpp)
OBJ_JOIN = $(patsubst $(SRC_DIR)/join/%.cpp, $(OBJ_DIR)/join/%.o, $(SRC_JOIN))
SRC_GROUP = $(wildcard $(SRC_DIR)/group/*.cpp)
OBJ_GROUP = $(patsubst $(SRC_DIR)/group/%.cpp, $(OBJ_DIR)/group/%.o, $(SRC_GROUP))
SRC_AGG = $(wildcard $(SRC_DIR)/aggregation/*.cpp)
OBJ_AGG = $(patsubst $(SRC_DIR)/aggregation/%.cpp, $(OBJ_DIR)/aggregation/%.o, $(SRC_AGG))
SRC_STARJOIN = $(wildcard $(SRC_DIR)/starjoin/*.cpp)
OBJ_STARJOIN = $(patsubst $(SRC_DIR)/starjoin/%.cpp, $(OBJ_DIR)/starjoin/%.o, $(SRC_STARJOIN))
SRC_OLAPCORE = $(wildcard $(SRC_DIR)/multi_compute_operator/*.cpp)
OBJ_OLAPCORE = $(patsubst $(SRC_DIR)/multi_compute_operator/%.cpp, $(OBJ_DIR)/multi_compute_operator/%.o, $(SRC_OLAPCORE))
SRC_GPUOLAPCORE = $(wildcard $(SRC_DIR)/gpu_multi_compute_operator/*.cu)
OBJ_GPUOLAPCORE = $(patsubst $(SRC_DIR)/gpu_multi_compute_operator/%.cu, $(OBJ_DIR)/gpu_multi_compute_operator/%.o, $(SRC_GPUOLAPCORE))
SRC_Q5 = $(wildcard $(SRC_DIR)/final_test/*.cpp)
OBJ_Q5 = $(patsubst $(SRC_DIR)/final_test/%.cpp, $(OBJ_DIR)/final_test/%.o, $(SRC_Q5))
ALL : OLAPcore

OLAPcore: $(OBJ_OLAPCORE)
	$(cc)  $<  $(CFLAGS)  -o   $@
$(OBJ_OLAPCORE): $(SRC_OLAPCORE)
	$(cc) -c $(CFLAGS)  $< -o $@

clean:
	rm -rf $(OBJ_SEL) $(OBJ_PRO) $(OBJ_JOIN) $(OBJ_GROUP)  $(OBJ_AGG)  $(OBJ_STARJOIN) $(OBJ_OLAPCORE) $(OBJ_Q5) select_test project_test join_test group_test starjoin_test OLAPcore GPUOLAPcore Q5

run: OLAPcore
	./OLAPcore --SF 1

.PHONY: clean ALL
