#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include <stdbool.h>
#include <time.h> // benchmark perf
#include <math.h>
#include <arm_neon.h> // optimization
#include <stdint.h>


// TODO: add unitary testing https://github.com/ThrowTheSwitch/Unity/tree/master
// benchmark tool on m4 max / m2 pro, ssh into i5-9600k to use AVX/SIMD intrinsics
// arm neon https://developer.arm.com/architectures/instruction-sets/intrinsics/



// allocates a tensor struct in memory
tensor_t* tensor_create(int ndim, size_t* shape) {
  
  // size
  int total_size = 1;
  for (int i = 0; i < ndim; i++) {
    total_size *= shape[i];
  }

	// calculate nbr of bytes for each part we allocate
	size_t size_struct = sizeof(tensor_t);
	size_t size_shape = ndim * sizeof(size_t);
	size_t size_stride = ndim * sizeof(int);
	size_t size_data = total_size * sizeof(float);

	// allocates an array of 1 object of total size and inits all bytes to zero
	// we use uint8_t for byte-level precision
	uint8_t* memory = (uint8_t*)calloc(1, size_struct + size_shape + size_stride + size_data);
	if (!memory) return NULL;


	// distributing the memory
	tensor_t* t = (tensor_t*)memory;


	t->shape = (size_t*)(memory + size_struct); 
	t->stride = (int*)(memory + size_struct + size_shape);
	t->data = (float*)(memory + size_struct + size_shape + size_stride);


  t->ndim = ndim;
	t->total_size = total_size;
	// copy shape 
  for (int i = 0; i < ndim; i++)
    t->shape[i] = shape[i];


		
	// calculate strides with row-major order
  t->stride[ndim-1] = 1;
  for (int i = ndim-2; i >= 0; i--)
    t->stride[i] = t->stride[i+1] * shape[i+1];



  return t;
} 


// frees this tensor
void tensor_free(tensor_t* t) {
  if (t == NULL) return;
  if (t->data != NULL) free(t->data);
  if (t->shape != NULL) free(t->shape);
  if (t->stride != NULL) free(t->stride);
  free(t);
}


void tensor_print(tensor_t* t) { 
  if (t == NULL || t->data == NULL) {
    printf("Tensor: NULL\n");
    return;
  }
  printf("Tensor(\n");
  printf("\tdata: ");
  for (int i = 0; i < t->total_size; i++)
    printf("%f,", ((float*)t->data)[i]); // casts a float pointer to the actual data
  printf("\n)\n");
}


// fills tensor with data from 0 to max size
void tensor_fill(tensor_t* t) {
  float* float_data = (float*)t->data;
  for (int i = 0; i < t->total_size; i++) {
    float_data[i] = (float)i;
  }
}

// different getters for each dimension
float tensor_get1d(tensor_t* t, int i) {
  if (t->ndim != 1 || i >= t->shape[0]) 
    return 0.0f;

  int idx = i;
  return ((float*)t->data)[idx];
}

float tensor_get2d(tensor_t* t, int i, int j) {
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
  if (t->ndim != 4 || i >= t->shape[0] || j >= t->shape[1] || k >= t->shape[2] || l >= t->shape[3])
    return 0.0f;

  int idx = i * t->stride[0] + j * t->stride[1] + k * t->stride[2] + l;
  return ((float*)t->data)[idx];

}

// different setters for each dimensions
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
  if (t->ndim != 4 || i >= t->shape[0] || j >= t->shape[1] || k >= t->shape[2] || l >= t->shape[3])
    return false;

  int idx = i * t->stride[0] + j * t->stride[1] + k * t->stride[2] + l;
  ((float*)t->data)[idx] = value;
  return true;
}




// Binary OPs /*---------------------------------------------------------------------------*/

tensor_t* tensor_add(tensor_t* a, tensor_t* b) {

	if (a->ndim != b->ndim) return NULL;
	for (int i = 0; i < a->ndim; i++) {
		if (a->shape[i] != b->shape[i]) return NULL;
	}

	tensor_t* t = tensor_create(a->ndim, a->shape);

	float* a_data = (float*)a->data;
	float* b_data = (float*)b->data;
	float* t_data = (float*)t->data;
	for (int i = 0; i < a->total_size; i++ ) {
		t_data[i] = a_data[i] + b_data[i];
	}
	return t; 
}

tensor_t* tensor_sub(tensor_t* a, tensor_t* b) {

	if (a->ndim != b->ndim) return NULL;
	for (int i = 0; i < a->ndim; i++) {
		if (a->shape[i] != b->shape[i]) return NULL;
	}

	tensor_t* t = tensor_create(a->ndim, a->shape);

	float* a_data = (float*)a->data;
	float* b_data = (float*)b->data;
	float* t_data = (float*)t->data;
	for (int i = 0; i < a->total_size; i++ ) {
		t_data[i] = a_data[i] -  b_data[i];
	}
	return t;
}

tensor_t* tensor_div(tensor_t* a, tensor_t* b) {

	if (a->ndim != b->ndim) return NULL;
	for (int i = 0; i < a->ndim; i++) {
		if (a->shape[i] != b->shape[i]) return NULL;
	}

	tensor_t* t = tensor_create(a->ndim, a->shape);

	float* a_data = (float*)a->data;
	float* b_data = (float*)b->data;
	float* t_data = (float*)t->data;
	for (int i = 0; i < a->total_size; i++ ) {
		t_data[i] = a_data[i] /  b_data[i];
	}
	return t;
}

tensor_t* tensor_mul(tensor_t* a, tensor_t* b) {

	if (a->ndim != b->ndim) return NULL;
	for (int i = 0; i < a->ndim; i++) {
		if (a->shape[i] != b->shape[i]) return NULL;
	}

	tensor_t* t = tensor_create(a->ndim, a->shape);

	float* a_data = (float*)a->data;
	float* b_data = (float*)b->data;
	float* t_data = (float*)t->data;
	for (int i = 0; i < a->total_size; i++ ) {
		t_data[i] = a_data[i] *  b_data[i];
	}

	return t;

}

tensor_t* matmul(tensor_t* a, tensor_t* b) {
	
	// A = m rows, n columns
	// B = n rows, p columns
	// C = m rows, p columns

	int m = a->shape[0]; // A's rows
	int n = a->shape[1]; // A's cols
	int p = b->shape[1]; // B's cols

	size_t c_shape[2] = { (size_t)m, (size_t)p };
	tensor_t* c = tensor_create(2, c_shape);
	
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

// Unary OPs /*---------------------------------------------------------------------------*/

typedef float (*unary_func_t)(float);

void tensor_apply_unary(tensor_t* t, unary_func_t func) {
	float* data = (float*)t->data;
	

	// pragma once simd 
	for (int i = 0; i < t->total_size; i++) {
		data[i] = func(data[i]);
	}
}


void tensor_log2(tensor_t* t) { tensor_apply_unary(t, log2f); } 
void tensor_exp2(tensor_t* t) { tensor_apply_unary(t, exp2f); } 
void tensor_sin(tensor_t* t) { tensor_apply_unary(t, sinf); } 
void tensor_sqrt(tensor_t* t) { tensor_apply_unary(t, sqrtf); }


// cant apply unary_func_t here
void tensor_neg(tensor_t* t) { 
	float* data = (float*)t->data;
	for (int i = 0; i < t->total_size; i++) {
		data[i] = -data[i];
	}
}



// main loop, testing -> will refactor to proper testing files
int
main (int argc, char* argv[]) {
	
	// lets assume square matrices
  size_t shape[] = {2,2}; 
  size_t shape1[] = {16,16};
  size_t shape2[] = {16,16};

	tensor_t* result = tensor_create(2, shape);

  tensor_t* a = tensor_create(2, shape1);
  tensor_t* b = tensor_create(2, shape2);

	tensor_fill(a);
	// tensor_print(a);
	tensor_fill(b);
	// tensor_print(b);

	tensor_fill(result);
	//	tensor_t* result = tensor_log2(t);
	

	tensor_neg(result);
	//	tensor_sqrt(result);
	//tensor_exp2(result);
	//tensor_log2(result);
	//tensor_sin(result);
	tensor_print(result);
	
  printf("Created tensor at %p\n", (void*)result);
  printf("Dimensions: %d\n", result->ndim);
  
  printf("Shape: [%zu, %zu]\n", result->shape[0], result->shape[1]);


 	tensor_free(a);
	tensor_free(b);
	tensor_free(result);


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





