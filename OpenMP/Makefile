ROOTS := $(filter-out result/ utilities/, $(wildcard */))
MAKES := $(wildcard $(addsuffix */Makefile,$(ROOTS))) $(wildcard $(addsuffix */*/Makefile,$(ROOTS)))
BENCHMARKS := $(sort $(dir $(MAKES)))

.PHONY: all $(BENCHMARKS) clean
all: $(BENCHMARKS)

$(BENCHMARKS):
	$(MAKE) -C $@

clean:
	for BENCHMARK in $(BENCHMARKS); do $(MAKE) -C $$BENCHMARK veryclean; done
