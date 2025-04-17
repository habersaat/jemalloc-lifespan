CC = gcc
CFLAGS = -I jemalloc/include
LDFLAGS = jemalloc/lib/libjemalloc.a

# Targets
TARGETS = test test_reclaimer test_misclassification test_promotion test_multi_promotion test_lifespan_stress

all: jemalloc $(TARGETS)

# Build jemalloc
jemalloc:
	cd jemalloc && \
	make clean && \
	./autogen.sh && \
	./configure --with-jemalloc-prefix=je_ --disable-shared --enable-debug && \
	make -j && \
	cd ..

# Build test executables
test: test.c jemalloc
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

test_reclaimer: test_reclaimer.c jemalloc
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

test_misclassification: test_misclassification.c jemalloc
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

test_promotion: test_promotion.c jemalloc
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

test_multi_promotion: test_multi_promotion.c jemalloc
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

test_lifespan_stress: test_lifespan_stress.c jemalloc
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

# Clean up
clean:
	rm -f $(TARGETS)
	cd jemalloc && make clean && cd ..

.PHONY: all jemalloc clean
