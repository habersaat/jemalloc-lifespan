## Compiling `test.c` with the locally modified jemalloc

First, build jemalloc. From the `jemalloc-lifespan/` directory, run:

```bash
cd jemalloc
make clean
./autogen.sh
./configure --with-jemalloc-prefix=je_ --enable-debug
make -j
cd ..
```

Connect to benchmark.py with:

```bash
source venv/bin/activate
LD_PRELOAD=./jemalloc/lib/libjemalloc.so venv/bin/python3 benchmark.py
```

Run the following from the root `jemalloc-lifespan/` directory:

```bash
gcc test.c -o test \
  -I jemalloc/include \
  jemalloc/lib/libjemalloc.a
```

```bash
gcc test_reclaimer.c -o test_reclaimer \
  -I jemalloc/include \
  jemalloc/lib/libjemalloc.a
```

```bash
gcc test_misclassification.c -o test_misclassification \
  -I jemalloc/include \
  jemalloc/lib/libjemalloc.a
```

```bash
gcc test_promotion.c -o test_promotion \
  -I jemalloc/include \
  jemalloc/lib/libjemalloc.a
```

```bash
gcc test_multi_promotion.c -o test_multi_promotion \
  -I jemalloc/include \
  jemalloc/lib/libjemalloc.a
```

```bash
gcc test_lifespan_stress.c -o test_lifespan_stress \
  -I jemalloc/include \
  jemalloc/lib/libjemalloc.a
```