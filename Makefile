# BdG Dipolar Library Makefile
CC = gcc
CFLAGS = -std=c11 -O3 -march=native -fopenmp -fPIC -Wall -Wextra

# Profiling: make PROFILE=1
ifeq ($(PROFILE),1)
  CFLAGS += -DBDG_PROFILE
endif

# BLAS backend: MKL (default) or OPENBLAS
BLAS_BACKEND ?= MKL

# Path to LOBPCG library
LOBPCG_DIR ?= $(HOME)/LOBPCG

ifeq ($(BLAS_BACKEND),MKL)
  BLAS_INC = $(MKLROOT)/include
  BLAS_LIB = $(MKLROOT)/lib/intel64
  BLAS_LINK = -L$(BLAS_LIB) -Wl,-rpath,$(BLAS_LIB) \
              -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm
  FFTW_INC = $(MKLROOT)/include/fftw
  FFTW_LINK =
  CFLAGS += -DUSE_MKL -Wno-incompatible-pointer-types
else ifeq ($(BLAS_BACKEND),OPENBLAS)
  BLAS_INC = /usr/include
  BLAS_LIB = /usr/lib
  BLAS_LINK = -lopenblas -lpthread -lm
  FFTW_INC = /usr/include
  FFTW_LINK = -lfftw3
endif

LOBPCG_INC = -I$(LOBPCG_DIR) -I$(LOBPCG_DIR)/include -I$(LOBPCG_DIR)/include/lobpcg
LOBPCG_LIB = $(LOBPCG_DIR)/build/liblobpcg.a

INCLUDES = -Iinclude -Isrc -I$(BLAS_INC) -I$(FFTW_INC) $(LOBPCG_INC)
LDFLAGS = $(LOBPCG_LIB) $(BLAS_LINK) $(FFTW_LINK)

# Sources (all .c in src/)
SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c,build/%.o,$(SRC))

# Auto-discover tests
TEST_SRC = $(wildcard tests/test_*.c)
TESTS = $(patsubst tests/%.c,build/%.ex,$(TEST_SRC))

# Examples
EXAMPLE_SRC = $(wildcard examples/*.c)
EXAMPLES = $(patsubst examples/%.c, build/ex_%.ex,$(EXAMPLE_SRC))

.PHONY: all lib tests run-tests examples clean

all: lib tests

lib: build build/libbdg.a

tests: build lib $(TESTS)

examples: build lib $(EXAMPLES)

build:
	@mkdir -p build

build/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

build/libbdg.a: $(OBJ)
	ar rcs $@ $(OBJ) 2>/dev/null || ar rcs $@

# Tests link against both libbdg and liblobpcg
build/%.ex: tests/%.c build/libbdg.a
	$(CC) $(CFLAGS) $(INCLUDES) $< build/libbdg.a -o $@ $(LDFLAGS)

# Examples link the same way
build/ex_%.ex: examples/%.c build/libbdg.a
	$(CC) $(CFLAGS) $(INCLUDES) $< build/libbdg.a -o $@ $(LDFLAGS)

run-tests: tests
	@for t in $(TESTS); do echo ">>> $$t"; $$t && echo "[PASS]" || echo "[FAIL]"; done

clean:
	rm -rf build $(OBJ) *~
