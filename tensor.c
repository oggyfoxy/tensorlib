
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h> 
#include <math.h>
#include <stdint.h>
#include <sys/mman.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h> // AVX2 (256 bits)
  #define SIMD_AVX
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
  #include <arm_neon.h>
  #define SIMD_NEON
#else 
  #define SIMD_NONE
#endif 



#include "tensor.h"

// benchmark tool on m4 max / m2 pro, ssh into i5-9600k to use AVX/SIMD intrinsics
// arm neon https://developer.arm.com/architectures/instruction-sets/intrinsics/


// allocates a tensor struct in memory (v2)
// TODO v3 add alignment, add lazy memory alloc (only on creation) 
tensor_t* tensor_create(int ndim, size_t* shape) {
  
 	// calculate nbr of bytes for each part we allocate
	size_t size_struct = sizeof(tensor_t);
	size_t size_shape = ndim * sizeof(size_t);
	size_t size_stride = ndim * sizeof(int);
	// size_t size_data = total_size * sizeof(float);

	// allocates an array of 1 object of total size and inits all bytes to zero
	// we use uint8_t for byte-level precision
	uint8_t* memory = (uint8_t*)calloc(1, size_struct + size_shape + 
                     size_stride);

	if (!memory) return NULL;

	// distributing the memory
	tensor_t* t = (tensor_t*)memory;
	t->shape = (size_t*)(memory + size_struct); 
	t->stride = (int*)(memory + size_struct + size_shape);
  t->ndim = ndim;
  t->realized = false;

  // size
  int total_size = 1;
  for (int i = 0; i < ndim; i++) {
    total_size *= shape[i];
    t->shape[i] = shape[i]; // copy shape 
  }
	t->total_size = total_size;

	// calculate strides with row-major order
  t->stride[ndim-1] = 1;
  for (int i = ndim-2; i >= 0; i--)
    t->stride[i] = t->stride[i+1] * shape[i+1];

  return t;
} 


void tensor_realize(tensor_t* t) {
  if (t->realized) return;
  t->data = mmap(NULL,
                  t->total_size * sizeof(float),
                  PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS,
                  -1, 0);
  t->realized = true;
}

// frees this tensor
void tensor_free(tensor_t* t) {
  if (t) free(t);
}


void tensor_print(tensor_t* t) { 
  if (t == NULL || t->data == NULL) {
    printf("tensor: NULL\n");
    return;
  }
  printf("\ttensor(");
  

//  for (int i = 0; i < t->ndim; i++) {
  printf("[");

  //}
  for (int i = 0; i < t->total_size; i++) {
    // casts a float pointer to the actual data   
    printf("%f", ((float*)t->data)[i]); 
    int closed = 0;
    for (int d = t->ndim-1; d >= 0; d--) {
      if ((i+1) % t->stride[d] == 0) {
        printf("]");
        closed++;
      } else { break; }

    }

    if (i < t->total_size - 1) {
      printf(", ");

      // for each opeing dims
      for (int d = t->ndim - closed; d < t->ndim; d++) {
        printf("[");
      }

      if (closed >= 2) {
        printf("\n "); 
      }
    }
  }
 
  
  printf("]\n)");
}


// fills tensor with data from 0 to max size
void tensor_fill(tensor_t* t) {
  float* float_data = (float*)t->data;
  for (int i = 0; i < t->total_size; i++) {
    float_data[i] = (float)i;
  }
}



// Getters /*------------------------------------------------------------------*/

float tensor_get1d(tensor_t* t, size_t i) {
  if (t->ndim != 1 || i >= t->shape[0]) 
    return 0.0f;

  int idx = i;
  return ((float*)t->data)[idx];
}

float tensor_get2d(tensor_t* t, size_t i, size_t j) {
  if (t->ndim != 2 || i >= t->shape[0] || j >= t->shape[1])
    return 0.0f;


   int idx = i * t->stride[0] + j;
   return ((float*)t->data)[idx];
}


float tensor_get3d(tensor_t* t, int i, int j, int k) {
  if (t->ndim != 3 || i >= t->shape[0] || j >= t->shape[1] || k >= t->shape[2])
    return 0.0f;

  int idx = i * t->stride[0] + j * t->stride[1] + k;
  return ((float*)t->data)[idx];

}

float tensor_get4d(tensor_t* t, int i, int j, int k, int l) {
  if (t->ndim != 4 || i >= t->shape[0] || j >= t->shape[1] 
      || k >= t->shape[2] || l >= t->shape[3])

    return 0.0f;

  int idx = i * t->stride[0] + j * t->stride[1] + k * t->stride[2] + l;
  return ((float*)t->data)[idx];

}

// TODO single getter for n dimensions
float tensor_get(tensor_t* t, size_t); 





// Setters /*------------------------------------------------------------------*/
bool tensor_set1d(tensor_t* t, int i, float value) {
  if (t->ndim != 1 || i >= t->shape[0]) 
    return false;

  int idx = i;
  ((float*)t->data)[idx] = value;
  return true;
}


bool tensor_set2d(tensor_t* t, int i, int j, float value) {
  if (t->ndim != 2 || i >= t->shape[0] || j >= t->shape[1])
    return false;

  int idx = i * t->stride[0] + j;
  ((float*)t->data)[idx] = value;
  return true;
}


bool tensor_set3d(tensor_t* t, int i, int j, int k, float value) {
  if (t->ndim != 3 || i >= t->shape[0] || j >= t->shape[1] || k >= t->shape[2])
    return false;

  int idx = i * t->stride[0] + j * t->stride[1] + k;
  ((float*)t->data)[idx] = value;
  return true;
}


bool tensor_set4d(tensor_t* t, int i, int j, int k, int l, float value) {
  if (t->ndim != 4 || i >= t->shape[0] || j >= t->shape[1]
      || k >= t->shape[2] || l >= t->shape[3])

    return false;

  int idx = i * t->stride[0] + j * t->stride[1] + k * t->stride[2] + l;
  ((float*)t->data)[idx] = value;
  return true;
}

//TODO single setter for n dimensions (probably need helper function)

bool tensor_set(tensor_t* t, int ndim, float value);

// Unary OPs /*----------------------------------------------------------------*/

// can optimize later
void tensor_apply_unary(tensor_t* t, unary_func_t func) {
	float* data = (float*)t->data;
	
	// pragma once simd 
	for (int i = 0; i < t->total_size; i++) {
		data[i] = func(data[i]);
	}
}


void tensor_log2_scalar(tensor_t* t) { tensor_apply_unary(t, log2f); } 
void tensor_exp2_scalar(tensor_t* t) { tensor_apply_unary(t, exp2f); } 
void tensor_sin_scalar(tensor_t* t) { tensor_apply_unary(t, sinf); } 
void tensor_sqrt_scalar(tensor_t* t) { tensor_apply_unary(t, sqrtf); }


// cant apply unary_func_t here
void tensor_neg_scalar(tensor_t* t) { 
	float* data = (float*)t->data;
  for (int i = 0; i < t->total_size; i++) {
		data[i] = -data[i];
	}
}

#ifdef SIMD_AVX_
  static void tensor_log2_avx2(tensor_t* t);
#endif




// Binary OPs /*---------------------------------------------------------------*/

void _op_add(float* a, float* b, float* out, int n) {
	for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}
void _op_sub(float* a, float* b, float* out, int n) {
	for (int i = 0; i < n; i++) out[i] = a[i] - b[i];
}
void _op_dot(float* a, float* b, float* out, int n) {
	for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}
void _op_idiv(float* a, float* b, float* out, int n) {
	for (int i = 0; i < n; i++) out[i] = a[i] / b[i];
}


// sanity checks

/*

void check_binaryop_data(tensor_t* lhs, tensor_t* rhs) {
  if (lhs->ndim != rhs->ndim) 
    printf("Tensors should have the same dimensions for add()");
  for (size_t i = 0; i < lhs->ndim; i++) {
    if (lhs->shape != rhs->shape)
      printf("Tensors should have the same shapes for add()");
  }
}
*/

tensor_t* tensor_apply_binary(tensor_t* a, tensor_t* b, binary_op_t op) {
	
	// check shapes 
	if (a->ndim != b->ndim) return NULL;
	for (int i = 0; i < a->ndim; i++) {
		if (a->shape[i] != b->shape[i]) return NULL;
	}
	
	// create result 
	tensor_t* t = tensor_create(a->ndim, a->shape);  
	if (!t) return NULL;  
  tensor_realize(t);
    
	// call kernel 
	op((float*)a->data, (float*)b->data, (float*)t->data, t->total_size); 
	
	return t;	
  
}


tensor_t* tensor_add(tensor_t* a, tensor_t* b) {
	return tensor_apply_binary(a, b, _op_add);
}

tensor_t* tensor_sub(tensor_t* a, tensor_t* b) {
	return tensor_apply_binary(a, b, _op_sub);
}

tensor_t* tensor_dot(tensor_t* a, tensor_t* b) {
	return tensor_apply_binary(a, b, _op_dot);
}

tensor_t* tensor_idiv(tensor_t* a, tensor_t* b) {
	return tensor_apply_binary(a, b, _op_idiv);
}




//TODO optimize matmul as much as needed (lean into GEMM / BLAS)
tensor_t* matmul(tensor_t* a, tensor_t* b) {
	
	// A = m rows, n columns
	// B = n rows, p columns
	// C = m rows, p columns

	int m = a->shape[0]; // A's rows
	int n = a->shape[1]; // A's cols
	int p = b->shape[1]; // B's cols

	size_t c_shape[2] = { (size_t)m, (size_t)p };
	tensor_t* c = tensor_create(2, c_shape);
  
  tensor_realize(c);

	float* a_data = (float*)a->data;
	float* b_data = (float*)b->data;
	float* c_data = (float*)c->data;
	
	// i = row index, j = col index, k = shared dims
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			float sum = 0.0f;	
			
			// shared dimension: n. k loops
			for (int k = 0; k < n; k++) {
				sum += a_data[i*n + k] * b_data[k*p + j]; // index = row * WIDTH + col
			}

			c_data[i*p + j] = sum;
		}
	}
	
	return c;

}



// if its contiguous return true, else false
// check if strides are contiguous between dimensions
bool is_contiguous(tensor_t* t) {
	int expected = 1; 
	for (int i = t->ndim-1; i >= 0; --i) {
		if (t->stride[i] != expected) { 
			return false;
		}
		expected *= (int)t->shape[i];
	}
	return true;
}



// main loop, testing -> will refactor to proper testing files

int
main (int argc, char* argv[]) {
	
	// lets assume square matrices
  size_t shape[] = {2,2}; 
  size_t shape1[] = {128,128};
  size_t shape2[] = {128,64};

	// tensor_t* result = tensor_create(2, shape);

  tensor_t* a = tensor_create(2, shape1);
  tensor_t* b = tensor_create(2, shape2);
  tensor_realize(a);
	tensor_fill(a);
	// tensor_print(a)

  tensor_realize(b);
	tensor_fill(b);
	

	// tensor_print(b);

	tensor_t* t = tensor_create(2,shape);
  tensor_realize(t);
	tensor_fill(t);
	printf("struct: %p\n", (void*)t);

	printf("data:   %p\n", t->data);
	printf("diff:   %ld bytes\n", (uint8_t*)t->data - (uint8_t*)t);



	tensor_t* result = matmul(a,b);
	//tensor_fill(result);
	

	//tensor_neg(result);
	//	tensor_sqrt(result);
	//tensor_exp2(result);
	//tensor_log2(result);
  //	tensor_sin(result);
	 tensor_print(result);
	
  printf("Created tensor at %p\n", (void*)result);
  printf("Dimensions: %d\n", result->ndim);
  
  printf("Shape: [%zu, %zu]\n", result->shape[0], result->shape[1]);


 	tensor_free(a);
	tensor_free(b);
	//	tensor_free(result);
	tensor_free(t);

  // bool one_d = tensor_set1d(t, 4, 29);
  // float two_d = tensor_get2d(t, 2, 1);
	//  float three_d = tensor_get3d(t, 0, 2, 3);
 	//  printf("element at 2d: %f\n", three_d);

	// tensor_print(t);

  // tests: free tensor

	// tensor_free(t);

  //t = NULL;
  // printf("Tensor freed\n");
  // printf("Tensor location after freeing: %p\n", (void*)t);
  return 0;
}



