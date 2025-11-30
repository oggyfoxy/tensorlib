

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "tensor.h"

#define EPS 1e-6f


//TODO: create more tests and stress-testing limitations. make simple

static void assert_float_eq(float a, float b, float eps, const char *msg) {
    if (fabsf(a - b) > eps) {
        fprintf(stderr, "ASSERT_FLOAT_EQ failed: %s (got %f, expected %f)\n",
                msg, a, b);
        assert(0);
    }
}

#define TEST(name) static void name(void)
#define RUN_TEST(name) \
    do { \
        name(); \
        printf("[PASS] %s\n", #name); \
    } while (0)

/* -------------------------------------------------------------------------- */

TEST(test_create_and_strides_2d) {
    size_t shape[2] = {2, 3};
    tensor_t *t = tensor_create(2, shape);
    assert(t != NULL);
    assert(t->ndim == 2);
    assert(t->total_size == 6);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);

    // row-major: [rows, cols] => stride = [cols, 1]
    assert(t->stride[1] == 1);
    assert(t->stride[0] == 3);

    assert(is_contiguous(t) && "newly created tensor should be contiguous");
    tensor_free(t);
}

TEST(test_fill_and_get2d) {
    size_t shape[2] = {2, 3};
    tensor_t *t = tensor_create(2, shape);
    tensor_fill(t);

    // tensor_fill writes data[i] = i
    assert_float_eq(tensor_get2d(t, 0, 0), 0.0f, EPS, "t[0,0]");
    assert_float_eq(tensor_get2d(t, 0, 1), 1.0f, EPS, "t[0,1]");
    assert_float_eq(tensor_get2d(t, 0, 2), 2.0f, EPS, "t[0,2]");
    assert_float_eq(tensor_get2d(t, 1, 0), 3.0f, EPS, "t[1,0]");
    assert_float_eq(tensor_get2d(t, 1, 1), 4.0f, EPS, "t[1,1]");
    assert_float_eq(tensor_get2d(t, 1, 2), 5.0f, EPS, "t[1,2]");

    tensor_free(t);
}

TEST(test_set_and_get1d) {
    size_t shape[1] = {4};
    tensor_t *t = tensor_create(1, shape);

    assert(tensor_set1d(t, 0, 10.0f));
    assert(tensor_set1d(t, 1, 20.0f));
    assert(tensor_set1d(t, 2, 30.0f));
    assert(tensor_set1d(t, 3, 40.0f));

    assert_float_eq(tensor_get1d(t, 0), 10.0f, EPS, "t[0]");
    assert_float_eq(tensor_get1d(t, 1), 20.0f, EPS, "t[1]");
    assert_float_eq(tensor_get1d(t, 2), 30.0f, EPS, "t[2]");
    assert_float_eq(tensor_get1d(t, 3), 40.0f, EPS, "t[3]");

    tensor_free(t);
}

TEST(test_binary_add) {
    size_t shape[1] = {4};
    tensor_t *a = tensor_create(1, shape);
    tensor_t *b = tensor_create(1, shape);

    tensor_fill(a); // [0,1,2,3]
    tensor_fill(b); // [0,1,2,3]

    tensor_t *c = tensor_add(a, b); // expect [0,2,4,6]
    float *cdata = (float *)c->data;

    assert_float_eq(cdata[0], 0.0f, EPS, "c[0]");
    assert_float_eq(cdata[1], 2.0f, EPS, "c[1]");
    assert_float_eq(cdata[2], 4.0f, EPS, "c[2]");
    assert_float_eq(cdata[3], 6.0f, EPS, "c[3]");

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

TEST(test_unary_neg) {
    size_t shape[1] = {3};
    tensor_t *t = tensor_create(1, shape);
    tensor_fill(t); // [0,1,2]

    tensor_neg(t);
    float *data = (float *)t->data;

    assert_float_eq(data[0], -0.0f, EPS, "neg[0]");
    assert_float_eq(data[1], -1.0f, EPS, "neg[1]");
    assert_float_eq(data[2], -2.0f, EPS, "neg[2]");

    tensor_free(t);
}

TEST(test_matmul_small) {
    // A: 2x3, B: 3x2 => C: 2x2
    size_t shapeA[2] = {2, 3};
    size_t shapeB[2] = {3, 2};

    tensor_t *A = tensor_create(2, shapeA);
    tensor_t *B = tensor_create(2, shapeB);

    float *a = (float *)A->data;
    float *b = (float *)B->data;

    // A =
    // [1 2 3
    //  4 5 6]
    a[0] = 1; a[1] = 2; a[2] = 3;
    a[3] = 4; a[4] = 5; a[5] = 6;

    // B =
    // [7  8
    //  9 10
    // 11 12]
    b[0] = 7;  b[1] = 8;
    b[2] = 9;  b[3] = 10;
    b[4] = 11; b[5] = 12;

    tensor_t *C = matmul(A, B);
    float *c = (float *)C->data;

    // Expected C =
    // [ 58  64
    //  139 154]
    assert_float_eq(c[0],  58.0f, EPS, "C[0,0]");
    assert_float_eq(c[1],  64.0f, EPS, "C[0,1]");
    assert_float_eq(c[2], 139.0f, EPS, "C[1,0]");
    assert_float_eq(c[3], 154.0f, EPS, "C[1,1]");

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(test_is_contiguous_false_when_stride_messed) {
    size_t shape[2] = {2, 3};
    tensor_t *t = tensor_create(2, shape);
    assert(is_contiguous(t));

    // Manually break contiguity by messing with stride
    t->stride[1] = 2; // last dim stride should be 1
    assert(!is_contiguous(t));

    tensor_free(t);
}


int main(void) {
    RUN_TEST(test_create_and_strides_2d);
    RUN_TEST(test_fill_and_get2d);
    RUN_TEST(test_set_and_get1d);
    RUN_TEST(test_binary_add);
    RUN_TEST(test_unary_neg);
    RUN_TEST(test_matmul_small);
    RUN_TEST(test_is_contiguous_false_when_stride_messed);

    printf("All tests passed.\n");
    return 0;
}

