# Makefile — gemma4 on colibri
#
# Everything here is overridable. The defaults auto-detect Intel vs Apple silicon
# and try to find Homebrew's libomp on macOS.
#
#   make                          # auto
#   make OMP=0                    # single-threaded (no OpenMP at all)
#   make METAL=0                  # pure-CPU binary (no Metal compiled in)
#   make CC=gcc-14 OMPFLAGS=-fopenmp OMPLIBS=-fopenmp     # Homebrew GCC
#   make ARCHFLAGS="-mcpu=apple-m3"                        # pin the CPU
#   make check                    # build + run the full regression suite

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CC      ?= cc
OPT     ?= -O3
WARN    ?= -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare

# ---- CPU features -----------------------------------------------------------
# q40.h has three kernels: AVX2, NEON (with an ARMv8.2 dotprod fast path), and a
# portable scalar fallback. Pick the right one or you silently get the slow path.
ifeq ($(UNAME_M),arm64)                 # Apple silicon
  ARCHFLAGS ?= -mcpu=apple-m1           # M1 baseline: implies NEON + dotprod
else ifeq ($(UNAME_M),aarch64)
  ARCHFLAGS ?= -march=armv8.2-a+dotprod
else                                    # x86_64 (Intel Mac, Linux)
  ARCHFLAGS ?= -march=native -mavx2 -mfma
endif

# ---- Metal ------------------------------------------------------------------
# Auto-enabled on macOS. METAL=0 builds a pure-CPU binary. Even with it compiled in,
# --no-metal disables it at runtime, and any Metal failure falls back to the CPU.
METAL ?= 1
ifeq ($(UNAME_S),Darwin)
  ifeq ($(METAL),1)
    METAL_CFLAGS ?= -DCOLI_METAL
    METAL_LDFLAGS ?= -framework Metal -framework Foundation
    METAL_OBJ ?= metal.o
  endif
endif

# ---- OpenMP -----------------------------------------------------------------
# Apple's clang needs Homebrew's libomp and the -Xpreprocessor dance. Homebrew GCC
# just takes -fopenmp. Override OMPFLAGS/OMPLIBS if this guess is wrong for you.
OMP ?= 1
ifeq ($(OMP),1)
  ifeq ($(UNAME_S),Darwin)
    # MacPorts first, then Homebrew. Note OMPFLAGS are COMPILE flags and OMPLIBS are
    # LINK flags -- passing an -I path as OMPLIBS compiles fine and then fails to link.
    MACPORTS_OMP := /opt/local
    ifeq ($(wildcard $(MACPORTS_OMP)/include/libomp/omp.h),$(MACPORTS_OMP)/include/libomp/omp.h)
      OMPFLAGS ?= -Xpreprocessor -fopenmp -I$(MACPORTS_OMP)/include/libomp
      OMPLIBS  ?= -L$(MACPORTS_OMP)/lib/libomp -lomp
    endif
    BREW_OMP := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(BREW_OMP),)
      OMPFLAGS ?= -Xpreprocessor -fopenmp -I$(BREW_OMP)/include
      OMPLIBS  ?= -L$(BREW_OMP)/lib -lomp
    endif
    OMPFLAGS ?= -fopenmp
    OMPLIBS  ?= -fopenmp
  else
    OMPFLAGS ?= -fopenmp
    OMPLIBS  ?= -fopenmp
  endif
endif

CFLAGS  ?= $(OPT) $(WARN) $(ARCHFLAGS) $(OMPFLAGS) $(METAL_CFLAGS) -I.
LDFLAGS ?= -lm -lpthread $(OMPLIBS) $(METAL_LDFLAGS)

all: gemma4

# -fno-objc-arc on purpose: ARC forbids Objective-C pointers as C-struct members, and
# the pointer->MTLBuffer map needs exactly that. Manual retain, and nothing is ever
# released -- every Metal object lives for the process.
metal.o: metal.mm gpu.h
	$(CC) $(OPT) $(METAL_CFLAGS) -I. -fno-objc-arc -c metal.mm -o $@

gemma4: gemma4.c q40.h g4tok.h kvq.h gpu.h openai_http.h openai_json.h $(METAL_OBJ)
	$(CC) $(CFLAGS) gemma4.c $(METAL_OBJ) -o $@ $(LDFLAGS)

# COLI_F32ACT keeps activations in f32 (weights stay q4_0). Slower; used only to
# separate the int8-activation approximation from an actual bug when validating.
gemma4-exact: gemma4.c q40.h g4tok.h kvq.h gpu.h $(METAL_OBJ)
	$(CC) $(CFLAGS) -DCOLI_F32ACT gemma4.c $(METAL_OBJ) -o $@ $(LDFLAGS)

test_q40: tests/test_q40.c q40.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_q40.c -o $@ -lm

test_kvq: tests/test_kvq.c kvq.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_kvq.c -o $@ -lm

# Simulates the Metal shader lane-for-lane on the CPU and diffs it against the
# reference. Validates the shader LOGIC (nibble order, the unaligned fp16 scale, the
# strided reduction) on any machine, Metal or not.
test_metal_sim: tests/test_metal_sim.c q40.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_metal_sim.c -o $@ -lm

# Full regression. Needs python3 + torch + transformers (for the fixture only).
check: gemma4 gemma4-exact test_q40 test_kvq test_metal_sim
	./test_q40
	./test_kvq
	./test_metal_sim
	python3 tools/convert_gemma4.py --fixture --ram 8 --ctx 64 /tmp/g4fix
	python3 tools/gemma4_check.py /tmp/g4fix
	./gemma4-exact /tmp/g4fix --check            # engine vs oracle, must be ~1e-7
	./gemma4-exact /tmp/g4fix --check --nobatch  # batch-union == sequential
	./gemma4       /tmp/g4fix --check            # int8-activation build
	./gemma4       /tmp/g4fix --check-gpu        # Metal vs CPU (no-op without Metal)
	python3 tools/gemma4_mtp_fixture.py /tmp/g4mtp
	python3 tools/convert_gemma4_mtp.py /tmp/g4mtp /tmp/g4fix
	python3 tools/gemma4_mtp_check.py /tmp/g4fix /tmp/g4mtp   # MTP head vs HF

clean:
	rm -f gemma4 gemma4-exact test_q40 test_kvq test_metal_sim metal.o

.PHONY: all check clean
