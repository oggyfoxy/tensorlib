
/*
 *   Tensor header, defining necessary structs to have a working tensor.
 *   A tensor is a multidimensional matrix holding elements of a single data type.
 *
 *   Author: @oggyfoxy
 *
 */


#include <stdbool.h>

#ifndef TENSOR_H
#define TENSOR_H

#define MAX_DIMS 4


// TODO: define datatype enum (placeholder)
typedef enum {
  INT8, // 
  FLOAT32,
  BFLOAT16,
  FLOAT16,
  // add more if necessary..
} dtype_t;

typedef enum {
  GPU,
  CPU,
} Device;


typedef struct {
  size_t* shape; // gets the number of dimensions (dynamic)
  void* data; // to not get stuck constrained in one type (you should type cast)
  // dtype_t dtype; // what type is our tensor: float64, float32, bf16, int8,
  //  uint8 etc. not really important as of now
  int* stride; // how many memory jumps to go to the next element in that dimension
  // stride(0) = row. stride(1) = column
  int total_size; // total nbr of elements
  int ndim;
  bool own_data; // does the tensor own this memory? 
} tensor_t;


// unary and binary abstractions
typedef float (*unary_func_t)(float);

typedef void (*binary_op_t)(float* a, float* b, float* out, int size);


// tensor interface
tensor_t* tensor_create(int ndim, size_t* shape); // done
void tensor_free(tensor_t* t); // done
void tensor_print(tensor_t* t); // done
void tensor_fill(tensor_t* t); // done

// TODO
// access ops (need to make general access ops)
// float tensor_get(tensor_t*, size_t* indices);
// bool tensor_set(tensor_t*, size_t* indices);


// Getters /*------------------------------------------------------------------*/

//TODO change int to size_t for getters and setters


float tensor_get1d(tensor_t* t, size_t i);
float tensor_get2d(tensor_t* t, size_t i, size_t j);
float tensor_get3d(tensor_t* t, int i, int j, int k);
float tensor_get4d(tensor_t* t, int i, int j, int k, int l);

bool is_contiguous(tensor_t* t);

// Setters /*------------------------------------------------------------------*/
bool tensor_set1d(tensor_t* t, int i, float value);
bool tensor_set2d(tensor_t* t, int i, int j, float value);
bool tensor_set3d(tensor_t* t, int i, int j, int k, float value);
bool tensor_set4d(tensor_t* t, int i, int j, int k, int l, float value);

// Unary OPs /*----------------------------------------------------------------*/

void tensor_apply_unary(tensor_t* t, unary_func_t func);

void tensor_exp2(tensor_t* t);
void tensor_log2(tensor_t* t);
void tensor_sqrt(tensor_t* t);
void tensor_sin(tensor_t* t);
void tensor_neg(tensor_t* t);


// Binary OPs /*---------------------------------------------------------------*/
tensor_t* tensor_apply_binary(tensor_t* a, tensor_t* b, binary_op_t op);

tensor_t* tensor_add(tensor_t* a, tensor_t* b); // returns a new tensor 
tensor_t* tensor_sub(tensor_t* a, tensor_t* b); // returns a new tensor 
tensor_t* tensor_idiv(tensor_t* a, tensor_t* b); // returns a new tensor 
tensor_t* tensor_dot(tensor_t* a, tensor_t* b); // returns a new tensor

// TODO bitwise ops (AND, OR, XOR, left-shift, right-shift
// (find correct abstraction first) 


// Movement OPs /*-------------------------------------------------------------*/


tensor_t* matmul(tensor_t* a, tensor_t* b); // naive implem works for ndim = 2.

/* TODO
 * reduction primitive: sum, mean, min/max, 
 * view / movement ops: reshape, permute (transpose), squeeze, slice 
 */ 


// matmul primitive  --> can build softmax, norms, linear layers, etc


#endif



