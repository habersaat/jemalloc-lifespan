## Compiling `test.c` with the locally modified jemalloc

First, build jemalloc. From the `jemalloc-lifespan/` directory, run:

```bash
cd jemalloc
make clean
./autogen.sh
./configure --with-jemalloc-prefix=je_ --disable-shared --enable-debug
make -j
cd ..
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