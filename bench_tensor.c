
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "tensor.h"

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);   // works on macOS 10.12+ and Linux
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#define BENCH(NAME, ITERS, CODE)                                \
    do {                                                        \
        uint64_t _start = now_ns();                             \
        for (size_t _i = 0; _i < (size_t)(ITERS); ++_i) {       \
            CODE;                                               \
        }                                                       \
        uint64_t _end = now_ns();                               \
        double total_ns = (double)(_end - _start);              \
        double ns_per   = total_ns / (double)(ITERS);           \
        printf("%-24s : %10.2f ns/op  (%zu iters)\n",           \
               (NAME), ns_per, (size_t)(ITERS));                \
    } while (0)

/* simple helpers to fill with something non-trivial */
static void fill_seq(tensor_t *t) {
    float *p = (float *)t->data;
    for (size_t i = 0; i < t->total_size; ++i) {
        p[i] = (float)i;
    }
}


// TODO: add more benchmarks, test stress limits

int main(void) {
    /* 1D add benchmark ----------------------------------------------------- */
    {
        size_t shape[1] = {1024};
        tensor_t *a = tensor_create(1, shape);
        tensor_t *b = tensor_create(1, shape);

        fill_seq(a);
        fill_seq(b);

        BENCH("add_1d_1024", 1000000, {
            tensor_t *c = tensor_add(a, b);
            tensor_free(c);
        });

        tensor_free(a);
        tensor_free(b);
    }

    /* 2D matmul benchmark -------------------------------------------------- */
    {
        size_t shapeA[2] = {128, 128};
        size_t shapeB[2] = {128, 128};

        tensor_t *A = tensor_create(2, shapeA);
        tensor_t *B = tensor_create(2, shapeB);

        fill_seq(A);
        fill_seq(B);

        BENCH("matmul_128x128", 100, {
            tensor_t *C = matmul(A, B);
            tensor_free(C);
        });

        tensor_free(A);
        tensor_free(B);
    }

    return 0;
}

