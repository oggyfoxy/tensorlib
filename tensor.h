/*

Tensor header, defining necessary structs to have a working tensor.
A tensor is a multidimensional matrix holding elements of a single data type.

Author: oggyfoxy

*/
#include <stdbool.h>
#ifndef TENSOR_H
#define TENSOR_H

#define MAX_DIMS 4


// TODO: define datatype enum (placeholder)
typedef enum {
  INT8, // 
  FLOAT32,
  // add more if necessary..
} dtype_t;

typedef enum {
  GPU,
  CPU,
} Device;


typedef struct {
  size_t* shape; // gets the number of dimensions (dynamic)
  void* data; // to not get stuck constrained in one type (you should type cast)
  // dtype_t dtype; // what type is our tensor: float64, float32, bf16, int8, uint8 etc. not really important as of now
  int* stride; // how many memory jumps to go to the next element in that dimension. stride(0) = row. stride(1) = column
  int total_size; // total nbr of elements
  int ndim;
} tensor_t;


// interface
tensor_t* tensor_create(int ndim, size_t* shape); // done
void tensor_free(tensor_t* t); // done
void tensor_print(tensor_t* t); // done
void tensor_fill(tensor_t* t); // done

// access ops (need to make general access ops)
// void* tensor_get(tensor_t*, size_t* indices);
// void tensor_set(tensor_t*, size_t* indices);

// getters 
float tensor_get1d(tensor_t* t, int i);
float tensor_get2d(tensor_t* t, int i, int j);
float tensor_get3d(tensor_t* t, int i, int j, int k);
float tensor_get4d(tensor_t* t, int i, int j, int k, int l);

// setters
bool tensor_set1d(tensor_t* t, int i, float value);
bool tensor_set2d(tensor_t* t, int i, int j, float value);
bool tensor_set3d(tensor_t* t, int i, int j, int k, float value);
bool tensor_set4d(tensor_t* t, int i, int j, int k, int l, float value);

// basic ops
tensor_t* tensor_add(tensor_t* a, tensor_t* b); // returns a new tensor 
tensor_t* tensor_sub(tensor_t* a, tensor_t* b); // returns a new tensor 
tensor_t* tensor_div(tensor_t* a, tensor_t* b); // returns a new tensor 
tensor_t* tensor_mul(tensor_t* a, tensor_t* b); // returns a new tensor



#endif
