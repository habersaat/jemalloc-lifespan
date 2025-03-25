## Compiling `test.c` with the locally modified jemalloc

Run the following from the root `jemalloc-lifespan/` directory:

```bash
gcc test.c -o test \
  -I jemalloc/include \
  jemalloc/lib/libjemalloc.a
```