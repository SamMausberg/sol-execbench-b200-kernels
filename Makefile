# SPDX-FileCopyrightText: 2026 sol-038-flux-rmsnorm contributors
# SPDX-License-Identifier: Apache-2.0

SHELL := /bin/bash
.DEFAULT_GOAL := help

EVALUATOR_REV := a9fa0804c793d438e70850c33fe34426e66d53dd
IMAGE := sol-execbench:$(EVALUATOR_REV)
TARGET ?= local

ROOT := $(abspath .)
EVALUATOR := $(ROOT)/third_party/sol-execbench
PROBLEM_DIR := $(ROOT)/.work/problem-038
STAGE_DIR := $(ROOT)/.work/stage/$(TARGET)
TRACE_DIR := $(ROOT)/.work/traces/$(TARGET)
PROFILE_DIR := $(ROOT)/.work/profiles/$(TARGET)
PACKAGE := /workspace/dist/038_flux_multi_head_rmsnorm_qk-b200.json

ifeq ($(TARGET),local)
  CLOCK_FLAG :=
  PROMOTE_FLAG :=
else ifeq ($(TARGET),b200)
  CLOCK_FLAG := --lock-clocks
  PROMOTE_FLAG := --promote /workspace/results/results.jsonl
else
  $(error TARGET must be local or b200)
endif

TOOL_RUN = docker run --rm \
	-v "$(ROOT):/workspace" \
	-w /workspace \
	--entrypoint python \
	"$(IMAGE)"

GPU_TOOL_RUN = docker run --rm --gpus all \
	-v "$(ROOT):/workspace" \
	-w /workspace \
	--entrypoint python \
	"$(IMAGE)"

GPU_EVAL_RUN = docker run --rm --gpus all \
	--ipc=host --privileged \
	--ulimit memlock=-1 --ulimit stack=67108864 \
	-v "$(ROOT):/workspace" \
	-v "$(EVALUATOR):/sol-execbench" \
	-v "$(ROOT)/.work/flashinfer-trace:/sol-execbench/data/flashinfer-trace" \
	-e FLASHINFER_TRACE_DIR=/sol-execbench/data/flashinfer-trace \
	-w /workspace \
	"$(IMAGE)"

.PHONY: help setup submodule image dirs fetch info lint stage test bench profile
.PHONY: status package verify-package

help:
	@sed -n 's/^## //p' Makefile

## make setup                 Initialize, build, fetch, and verify all locks.
setup: submodule image dirs
	$(TOOL_RUN) /workspace/tools/repo.py fetch
	$(TOOL_RUN) /workspace/tools/repo.py lint

submodule:
	git submodule update --init --recursive third_party/sol-execbench
	@test "$$(git -C third_party/sol-execbench rev-parse HEAD)" = "$(EVALUATOR_REV)" || \
		{ echo "evaluator revision does not match benchmark.lock.json" >&2; exit 1; }

image: submodule
	@docker image inspect "$(IMAGE)" >/dev/null 2>&1 || { \
		set -eu; \
		mkdir -p .work; \
		dockerfile=".work/evaluator.Dockerfile"; \
		sed 's#        useradd -m -u $${HOST_UID} -g $${HOST_GID} -s /bin/bash $${HOST_USER}; \\#        (id -u "$${HOST_USER}" >/dev/null 2>\&1 || useradd -m -u $${HOST_UID} -g $${HOST_GID} -s /bin/bash $${HOST_USER}); \\#' \
			third_party/sol-execbench/docker/Dockerfile > "$$dockerfile"; \
		if docker buildx version >/dev/null 2>&1; then \
			export DOCKER_BUILDKIT=1; \
		else \
			sed -E '/^[[:space:]]+--mount=/d; s/^RUN --mount=[^ ]+ \\/RUN \\/' \
				"$$dockerfile" > "$$dockerfile.legacy"; \
			mv "$$dockerfile.legacy" "$$dockerfile"; \
		fi; \
		build_user="$$(id -un)"; \
		base_user="$$(docker run --rm --entrypoint getent \
			nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 \
			passwd "$$(id -u)" 2>/dev/null | cut -d: -f1 || true)"; \
		if test -n "$$base_user"; then build_user="$$base_user"; fi; \
		docker build -t "$(IMAGE)" \
			--build-arg HOST_UID="$$(id -u)" \
			--build-arg HOST_GID="$$(id -g)" \
			--build-arg HOST_USER="$$build_user" \
			-f "$$dockerfile" third_party/sol-execbench; \
	}

dirs:
	mkdir -p .work/flashinfer-trace .work/data .work/problem-038

fetch: image dirs
	$(TOOL_RUN) /workspace/tools/repo.py fetch

## make info                  Print GPU, driver, architecture, and lock revisions.
info: image
	$(GPU_TOOL_RUN) /workspace/tools/repo.py info

## make lint                  Validate repository structure, locks, and manifests.
lint: submodule image
	$(TOOL_RUN) /workspace/tools/repo.py lint

stage: fetch
	$(TOOL_RUN) /workspace/tools/repo.py stage \
		--target "$(TARGET)" --output "/workspace/.work/stage/$(TARGET)"

## make test                  Compile locally and require all 16 workloads to pass.
## make test TARGET=b200      Compile for sm_100a and test all workloads on B200.
test: stage
	mkdir -p "$(TRACE_DIR)"
	$(GPU_EVAL_RUN) sol-execbench "/workspace/.work/problem-038" \
		--solution "/workspace/.work/stage/$(TARGET)/solution.json" \
		--compile-timeout 300 --timeout 1800 $(CLOCK_FLAG) \
		-o "/workspace/.work/traces/$(TARGET)/test.jsonl"

## make bench                 Run three local official-harness trials.
## make bench TARGET=b200     Run and promote three B200 trials.
bench: stage
	mkdir -p "$(TRACE_DIR)" "$(ROOT)/.work/summaries"
	@set -eu; for trial in 1 2 3; do \
		echo "official harness trial $$trial of 3"; \
		$(GPU_EVAL_RUN) sol-execbench "/workspace/.work/problem-038" \
			--solution "/workspace/.work/stage/$(TARGET)/solution.json" \
			--compile-timeout 300 --timeout 1800 $(CLOCK_FLAG) \
			-o "/workspace/.work/traces/$(TARGET)/trial-$$trial.jsonl"; \
	done
	$(TOOL_RUN) /workspace/tools/repo.py summarize \
		--target "$(TARGET)" \
		--output "/workspace/.work/summaries/$(TARGET).json" \
		$(PROMOTE_FLAG) \
		"/workspace/.work/traces/$(TARGET)/trial-1.jsonl" \
		"/workspace/.work/traces/$(TARGET)/trial-2.jsonl" \
		"/workspace/.work/traces/$(TARGET)/trial-3.jsonl"

## make profile WORKLOAD=0    Profile one workload with Nsight Compute if installed.
profile: stage
	@test -n "$(WORKLOAD)" || { echo "set WORKLOAD to an index from 0 to 15" >&2; exit 2; }
	mkdir -p "$(PROFILE_DIR)" "$(ROOT)/.work/profile-problem"
	$(TOOL_RUN) /workspace/tools/repo.py select-workload \
		--index "$(WORKLOAD)" \
		--problem "/workspace/.work/problem-038" \
		--output "/workspace/.work/profile-problem"
	$(GPU_EVAL_RUN) bash -lc 'command -v ncu >/dev/null || \
		{ echo "ncu is not installed in the pinned image" >&2; exit 2; }; \
		report="/workspace/.work/profiles/$(TARGET)/workload-$(WORKLOAD)"; \
		set +e; \
		ncu --target-processes all --set basic --launch-count 1 --kill 1 \
		--kernel-name "regex:rmsnorm_qk_.*_kernel" --force-overwrite \
		-o "$$report" \
		sol-execbench /workspace/.work/profile-problem \
		--solution /workspace/.work/stage/$(TARGET)/solution.json \
		--compile-timeout 300 --timeout 1800 $(CLOCK_FLAG); \
		profile_status=$$?; set -e; \
		test -s "$$report.ncu-rep" || exit "$$profile_status"; \
		echo "profile written to $$report.ncu-rep"'

## make status                Print the live kernel #38 leaderboard.
status: image
	$(TOOL_RUN) /workspace/tools/repo.py status

## make package               Build deterministic self-contained B200 JSON.
package: image
	$(TOOL_RUN) /workspace/tools/repo.py package --output "$(PACKAGE)"

## make verify-package        Rebuild and test the exact package on B200.
verify-package: package fetch
	$(TOOL_RUN) /workspace/tools/repo.py verify-package --artifact "$(PACKAGE)"
	$(GPU_EVAL_RUN) sol-execbench "/workspace/.work/problem-038" \
		--solution "/workspace/dist/038_flux_multi_head_rmsnorm_qk-b200.json" \
		--compile-timeout 300 --timeout 1800 --lock-clocks \
		-o "/workspace/.work/traces/b200/package.jsonl"
