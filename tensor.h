/*

Tensor header, defining necessary structs to have a working tensor.
A tensor is a multidimensional matrix holding elements of a single data type.

Author: oggyfoxy

*/

#ifndef TENSOR_H
#define TENSOR_H

#define MAX_DIMS 4


// TODO: define datatype enum (placeholder)
typedef enum {
  INT8, // 
  FLOAT32,
  // add more if necessary..
} dtype_t;


typedef struct {
  int* shape; // gets the number of dimensions (dynamic)
  void* data; // to not get stuck constrained in one type
  // dtype_t dtype; // what type is our tensor: float64, float32, bf16, int8, uint8 etc. not really important as of now
  int* stride; // how many memory jumps to go to the next element in that dimension. stride(0) = row. stride(1) = column
  int total_size; // total nbr of elements
  int ndim;
} tensor_t;


// interface
tensor_t* tensor_create(int ndim, int* shape);
void free_tensor(tensor_t* t);
void print_tensor(tensor_t* t);
void fill_tensor(tensor_t* t);

// access ops

// basic ops
tensor_t* tensor_add(tensor_t* a, tensor_t* b); // returns a new tensor 
tensor_t* tensor_sub(tensor_t* a, tensor_t* b); // returns a new tensor

#endif
