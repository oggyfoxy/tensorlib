/*

Tensor header, defining necessary structs to have a working tensor.
A tensor is a multidimensional matrix holding elements of a single data type.

520KB internal SRAM
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

// not making the shape and stride dynamically allocated since risky for embedded

typedef struct {
  int shape[MAX_DIMS]; // gets the number of dimensions
  void* data; // to not get stuck constrained in one type
  dtype_t dtype; // what type is our tensor: float64, float32, bf16, int8, uint8 etc. not really important as of now
  int stride[MAX_DIMS]; // how many memory jumps to go to the next element in that dimension. stride(0) = row. stride(1) = column
  int total_size; // total nbr of elements
  int ndim;
} tensor_t;



#endif
