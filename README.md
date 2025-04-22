## Video Demo

https://www.youtube.com/watch?v=VXPLSyxxCYo

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

NOT SURE IF WE'LL USE THIS YET BC PYMALLOC SUCKS
Connect to benchmark.py with:

```bash
source venv/bin/activate
LD_PRELOAD=./jemalloc/lib/libjemalloc.so venv/bin/python3 benchmark.py
```

To run correctness tests, run from the `jemalloc-lifespan/` directory:

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



ML Stuff I'm working on
- First run of synthetic_benchmark run with lifetime_ml_enabled as false
- then train using the lstm model
- finally, run synthetic_benchmark with lifetime_ml_enabled as true

```bash
gcc synthetic_benchmark.c -o synthetic_benchmark   -I jemalloc/include   jemalloc/lib/libjemalloc.a
./synthetic_benchmark
python3 train_lifetime_lstm.py
```
SET lifetime_ml_enabled TO TRUE
```bash
gcc synthetic_benchmark.c -o synthetic_benchmark   -I jemalloc/include   jemalloc/lib/libjemalloc.a
./synthetic_benchmark
```
