SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

LEONARDO_CUDA_PRESETS := leonardo-cuda-mpi leonardo-cuda-nccl leonardo-cuda-nvshmem leonardo-oshmpi
LEONARDO_SYCL_PRESETS := leonardo-sycl-mpi leonardo-sycl-oneccl leonardo-sycl-oneccl-oshmpi
LEONARDO_PRESETS := $(LEONARDO_CUDA_PRESETS) $(LEONARDO_SYCL_PRESETS)

.PHONY: help init bootstrap submit configure build clean leonardo leonardo-cuda leonardo-sycl leonardo-clean

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make init                       Bootstrap dependencies and build every' \
	  '                                  Leonardo preset' \
	  '  make bootstrap                  Build oneCCL and OSHMPI dependencies plus' \
	  '                                  leonardo-sycl-oneccl-oshmpi. TARGETS=...,' \
	  '                                  see cluster/leonardo/bootstrap.sh --list' \
	  '  make submit                     Submit experiments (BENCH=allreduce ...)' \
	  '  make configure PRESET=<preset>  Configure one CMake preset' \
	  '  make build PRESET=<preset>      Build one CMake preset' \
	  '  make clean PRESET=<preset>      Remove one preset or group build directory' \
	  '                                  Groups: leonardo, leonardo-cuda, leonardo-sycl' \
	  '  make leonardo                   Build all Leonardo presets' \
	  '  make leonardo-cuda              Build Leonardo CUDA-stack presets' \
	  '  make leonardo-sycl              Build Leonardo SYCL-stack presets' \
	  '  make leonardo-clean             Remove Leonardo build directories'

init:
	$(MAKE) bootstrap TARGETS=
	$(MAKE) leonardo

bootstrap:
	./cluster/leonardo/bootstrap.sh $(TARGETS)

submit:
	cluster/harness/launch.sh --all $(BENCH)

configure:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	cmake --preset "$(PRESET)"

build:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	cmake --build --preset "$(PRESET)"

clean:
	@test -n "$(PRESET)" || { echo 'missing PRESET=<preset>'; exit 2; }
	# build*/ rather than build/: selecting a non-default library version puts
	# that build in its own tree (build-nvshmem-3.7.2/, see
	# cluster/leonardo/layout.sh), and cleaning a preset should not leave the
	# other selections of it behind. The glob is at the top level only, so it
	# cannot catch a preset whose name merely extends this one.
	case "$(PRESET)" in
	  leonardo) presets="$(LEONARDO_PRESETS)" ;;
	  leonardo-cuda) presets="$(LEONARDO_CUDA_PRESETS)" ;;
	  leonardo-sycl) presets="$(LEONARDO_SYCL_PRESETS)" ;;
	  *) presets="$(PRESET)" ;;
	esac
	shopt -s nullglob
	paths=()
	for preset in $${presets}; do paths+=(build*/"$${preset}"); done
	if [[ $${#paths[@]} -eq 0 ]]; then echo 'Nothing to remove.'; exit 0; fi
	printf 'Removing:%s\n' " $${paths[*]}"
	rm -rf "$${paths[@]}"

leonardo: leonardo-cuda leonardo-sycl

leonardo-cuda:
	source cluster/leonardo/environment.sh cuda
	for preset in $(LEONARDO_CUDA_PRESETS); do
	  cmake --preset "$$preset"
	  cmake --build --preset "$$preset"
	done

leonardo-sycl:
	source cluster/leonardo/environment.sh sycl
	for preset in $(LEONARDO_SYCL_PRESETS); do
	  cmake --preset "$$preset"
	  cmake --build --preset "$$preset"
	done

leonardo-clean:
	$(MAKE) clean PRESET=leonardo
