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

all: gemma4 lfm25 maple ling

# -fno-objc-arc on purpose: ARC forbids Objective-C pointers as C-struct members, and
# the pointer->MTLBuffer map needs exactly that. Manual retain, and nothing is ever
# released -- every Metal object lives for the process.
metal.o: metal.mm gpu.h
	$(CC) $(OPT) $(METAL_CFLAGS) -I. -fno-objc-arc -c metal.mm -o $@

gemma4: gemma4.c q40.h g4tok.h kvarn.h gpu.h openai_http.h openai_json.h $(METAL_OBJ)
	$(CC) $(CFLAGS) gemma4.c $(METAL_OBJ) -o $@ $(LDFLAGS)

# COLI_F32ACT keeps activations in f32 (weights stay q4_0). Slower; used only to
# separate the int8-activation approximation from an actual bug when validating.
gemma4-exact: gemma4.c q40.h g4tok.h kvarn.h gpu.h $(METAL_OBJ)
	$(CC) $(CFLAGS) -DCOLI_F32ACT gemma4.c $(METAL_OBJ) -o $@ $(LDFLAGS)

# LFM2.5-8B-A1B. Metal is compiled in (the backend has a q8_0 kernel alongside the
# q4_0 one, which lfm25's apex gradient needs) but stays OFF unless you pass --metal:
# at 1.5 B active params over ~5.9 MiB experts, dispatch latency usually beats the
# arithmetic saved. See the comment on g_use_gpu in lfm25.c.
LFM_DEPS = lfm25.c q40.h lfmtok.h kvarn.h gpu.h openai_http.h openai_json.h
LFM_CFLAGS = $(OPT) $(WARN) $(ARCHFLAGS) $(OMPFLAGS) $(METAL_CFLAGS) -I.
LFM_LDFLAGS = -lm -lpthread $(OMPLIBS) $(METAL_LDFLAGS)

lfm25: $(LFM_DEPS) $(METAL_OBJ)
	$(CC) $(LFM_CFLAGS) lfm25.c $(METAL_OBJ) -o $@ $(LFM_LDFLAGS)

# COLI_F32ACT keeps activations in f32 (weights stay quantised). Slower; used only
# to separate the int8-activation approximation from an actual bug when validating.
lfm25-exact: $(LFM_DEPS) $(METAL_OBJ)
	$(CC) $(LFM_CFLAGS) -DCOLI_F32ACT lfm25.c $(METAL_OBJ) -o $@ $(LFM_LDFLAGS)

# Maple. Metal is compiled in (the backend has tq2 and q4a kernels for this model's
# two formats) but stays off unless you pass --metal; see g_use_gpu in maple.c for
# why, and for where it does pay.
MAPLE_DEPS = maple.c q40.h tq2.h gpu.h lfmtok.h kvarn.h openai_http.h openai_json.h
MAPLE_CFLAGS = $(OPT) $(WARN) $(ARCHFLAGS) $(OMPFLAGS) $(METAL_CFLAGS) -I.
MAPLE_LDFLAGS = -lm -lpthread $(OMPLIBS) $(METAL_LDFLAGS)

maple: $(MAPLE_DEPS) $(METAL_OBJ)
	$(CC) $(MAPLE_CFLAGS) maple.c $(METAL_OBJ) -o $@ $(MAPLE_LDFLAGS)

# COLI_F32ACT keeps activations in f32 (weights stay ternary). Slower; used only to
# separate the int8-activation approximation from an actual bug when validating.
maple-exact: $(MAPLE_DEPS) $(METAL_OBJ)
	$(CC) $(MAPLE_CFLAGS) -DCOLI_F32ACT maple.c $(METAL_OBJ) -o $@ $(MAPLE_LDFLAGS)

# Ling-3.0-tiny. Same two block formats as lfm25, so metal.mm needs nothing new;
# Metal is compiled in but stays off unless you pass --metal.
LING_DEPS = ling.c q40.h lfmtok.h kvarn.h gpu.h openai_http.h openai_json.h
LING_CFLAGS = $(OPT) $(WARN) $(ARCHFLAGS) $(OMPFLAGS) $(METAL_CFLAGS) -I.
LING_LDFLAGS = -lm -lpthread $(OMPLIBS) $(METAL_LDFLAGS)

ling: $(LING_DEPS) $(METAL_OBJ)
	$(CC) $(LING_CFLAGS) ling.c $(METAL_OBJ) -o $@ $(LING_LDFLAGS)

# COLI_F32ACT keeps activations in f32 (weights stay quantised). Slower; used only to
# separate the int8-activation approximation from an actual bug when validating.
ling-exact: $(LING_DEPS) $(METAL_OBJ)
	$(CC) $(LING_CFLAGS) -DCOLI_F32ACT ling.c $(METAL_OBJ) -o $@ $(LING_LDFLAGS)

test_tq2: tests/test_tq2.c tq2.h q40.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_tq2.c -o $@ -lm

test_lfmtok: tests/test_lfmtok.c lfmtok.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_lfmtok.c -o $@ -lm

test_q40: tests/test_q40.c q40.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_q40.c -o $@ -lm

test_kvarn: tests/test_kvarn.c kvarn.h q40.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_kvarn.c -o $@ -lm

# Simulates the Metal shader lane-for-lane on the CPU and diffs it against the
# reference. Validates the shader LOGIC (nibble order, the unaligned fp16 scale, the
# strided reduction) on any machine, Metal or not.
test_metal_sim: tests/test_metal_sim.c q40.h tq2.h
	$(CC) $(OPT) $(ARCHFLAGS) -I. tests/test_metal_sim.c -o $@ -lm

# Full regression. Needs python3 + torch + transformers (for the fixture only).
check: gemma4 gemma4-exact test_q40 test_kvarn test_metal_sim
	./test_q40
	./test_kvarn
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

# lfm25 regression: the tokenizer against HF (or the in-file reference), and the
# engine against a numpy oracle run on the DEQUANTISED container weights.
#
# DSpark is deliberately NOT here: the fixture carries no drafter, and what is worth
# checking needs the real 5 GB container. Run it by hand after touching the conv
# snapshot or the KVarN confirm logic, the two things speculation can silently
# corrupt. There is no exact-match assertion to make -- the verify is batched and
# decode is not, so greedy flips on near-ties -- but these two must diverge from the
# baseline at the SAME token, since a rewind bug would depend on the block size:
#   ./lfm25 ./lfm-ct --kv off --temp 0 --max_tokens 96 PROMPT
#   ./lfm25 ./lfm-ct --kv off --temp 0 --max_tokens 96 --dspark --ndraft 2 PROMPT
#   ./lfm25 ./lfm-ct --kv off --temp 0 --max_tokens 96 --dspark PROMPT
# PYTHON is overridable: these need numpy, and lfmtok_check wants `tokenizers` for
# an authoritative comparison (it falls back to a bundled reference without it).
#   make check-lfm25 PYTHON=.venv/bin/python
PYTHON ?= python3
LFMFIX ?= /tmp/lfmfix

check-lfm25: lfm25 lfm25-exact test_lfmtok
	$(PYTHON) tools/lfmtok_check.py ./lfm ./lfm-ct/tok.bin
	$(PYTHON) tools/convert_lfm25.py --fixture --ctx 64 --ram 8 --expert-edge 1 $(LFMFIX)
	$(PYTHON) tools/lfm25_oracle.py $(LFMFIX)
	./lfm25-exact $(LFMFIX) --check            # engine vs oracle, must be ~1e-6
	./lfm25-exact $(LFMFIX) --check --nobatch  # batched == sequential
	./lfm25       $(LFMFIX) --check            # int8-activation build
	./lfm25-exact $(LFMFIX) --check --pin 3    # pinning must not change the logits
	./lfm25       $(LFMFIX) --check-gpu        # Metal vs CPU (no-op without Metal)
	./lfm25       $(LFMFIX) --check --metal    # engine with the GPU path enabled

# maple regression: the ternary/q4a kernels against an independent reference, the
# tokenizer against HF, and the engine against a numpy oracle run on the DEQUANTISED
# container weights.
#
# tools/maple_mlx_check.py is deliberately NOT here: it needs mlx-lm and a Metal GPU,
# and it compares against the real 5 GB checkpoint. Run it by hand once after any
# change to the architecture -- it is the only check that would catch a systematic
# misreading of maple.py, which the oracle shares by construction.
MAPFIX ?= /tmp/mapfix

check-maple: maple maple-exact test_tq2 test_lfmtok
	./test_tq2
	$(PYTHON) tools/convert_maple.py --fixture --ctx 64 --ram 8 --verify $(MAPFIX)
	$(PYTHON) tools/maple_oracle.py $(MAPFIX)
	./maple-exact $(MAPFIX) --check            # engine vs oracle, must be ~1e-7
	./maple-exact $(MAPFIX) --check --nobatch  # batch-union == sequential
	./maple-exact $(MAPFIX) --check --batch 3  # and at a batch that straddles chunks
	./maple-exact $(MAPFIX) --check --pin 3    # pinning must not change the logits
	./maple       $(MAPFIX) --check            # int8-activation build
	./maple       $(MAPFIX) --check-gpu        # Metal vs CPU (no-op without Metal)
	./maple       $(MAPFIX) --check --metal    # engine with the GPU path enabled

# ling regression: the engine against a numpy oracle run on the DEQUANTISED container
# weights, plus the tokenizer against HF.
#
# tools/ling_hf_check.py is deliberately NOT here: it needs torch + transformers +
# fla-core and the real 8 GB checkpoint. Run it by hand once after any change to the
# architecture -- it is the only check that would catch a systematic misreading of the
# KDA kernel, which the oracle shares by construction.
LINGFIX ?= /tmp/lingfix

check-ling: ling ling-exact test_lfmtok
	$(PYTHON) tools/convert_ling.py --fixture --ctx 64 --ram 8 $(LINGFIX)
	$(PYTHON) tools/ling_oracle.py $(LINGFIX)
	./ling-exact $(LINGFIX) --check            # engine vs oracle, must be ~1e-6
	./ling-exact $(LINGFIX) --check --nobatch  # batch-union == sequential
	./ling-exact $(LINGFIX) --check --pin 3    # pinning must not change the logits
	./ling       $(LINGFIX) --check            # int8-activation build
	./ling       $(LINGFIX) --check-gpu        # Metal vs CPU (no-op without Metal)
	./ling       $(LINGFIX) --check --metal    # engine with the GPU path enabled
	@# The tokenizer needs the real checkpoint (the fixture has no vocabulary), so it
	@# is checked only when ./ling-tiny-fp8 is present rather than failing the suite.
	@if [ -f ./ling-tiny-fp8/tokenizer.json ]; then \
	   $(PYTHON) tools/convert_lfm_tokenizer.py ./ling-tiny-fp8/tokenizer.json $(LINGFIX)/tok.bin && \
	   $(PYTHON) tools/lfmtok_check.py ./ling-tiny-fp8 $(LINGFIX)/tok.bin; \
	 else echo "skipping the tokenizer check: no ./ling-tiny-fp8"; fi

clean:
	rm -f gemma4 gemma4-exact lfm25 lfm25-exact maple maple-exact ling ling-exact \
	      test_q40 test_kvarn test_metal_sim test_lfmtok test_tq2 metal.o

.PHONY: all check check-lfm25 check-maple check-ling clean
