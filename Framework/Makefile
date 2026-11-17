PYTHON ?= python3
CXX ?= g++
CC ?= gcc
OPENMP_FLAGS ?= -fopenmp
BUILD_DIR ?= build
REPO_ROOT := $(abspath ..)

.PHONY: all native validation check clean
all: native
native: validation $(BUILD_DIR)/window_dp $(BUILD_DIR)/markov_sampler

$(BUILD_DIR):
	mkdir -p $@

validation: | $(BUILD_DIR)
	PYTHONPATH=$(REPO_ROOT) $(PYTHON) -B -c "from Experiments.backends import ensure_binary; [ensure_binary(name, version, '$(abspath $(BUILD_DIR))') for name, version in [('speck32',64),('simon32',64),('simeck32',64),('cham64',128),('present80',80),('knot256',0)]]"

$(BUILD_DIR)/window_dp: aggregation/window_dp.c aggregation/simon_transition.h | $(BUILD_DIR)
	$(CC) -O3 $(OPENMP_FLAGS) $< -lm -o $@

$(BUILD_DIR)/markov_sampler: aggregation/markov_sampler.c aggregation/simon_transition.h | $(BUILD_DIR)
	$(CC) -O3 $(OPENMP_FLAGS) $< -lm -o $@

check:
	cd $(REPO_ROOT) && $(PYTHON) -B -m Experiments.checks --suite interfaces framework

clean:
	rm -rf $(BUILD_DIR)
