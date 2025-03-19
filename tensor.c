#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

// allocates a tensor struct in memory
tensor_t* create_tensor(int ndim, int* shape) {
  tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t)); // allocate memory for a tensor on the heap
  t->ndim = ndim;
  
  // allocate shape array
  t->shape = (int*)malloc(ndim * sizeof(int));
  
  // check if shape allocation failed
  if (!t->shape) {
    free(t);
    return NULL;
  }

  for (int i = 0; i < ndim; i++)
    t->shape[i] = shape[i];


  // allocate stride array
  t->stride = (int*)malloc(ndim * sizeof(int));

  if (!t->stride) {
    free(t);
    return NULL;
  }

  // calculate strides with row-major order
  t->stride[ndim-1] = 1;
  for (int i = ndim-2; i >= 0; i--)
    t->stride[i] = t->stride[i+1] * shape[i+1];

  // size
  t->total_size = 1;
  for (int i = 0; i < ndim; i++) {
    t->total_size *= shape[i];
  }

  // allocate data in memory and set it to 0
  t->data = calloc(t->total_size, sizeof(float)); // use float as implicit default

  // check if allocation failed
  if (!t->data) {
    free(t->stride);
    free(t->shape);
    free(t);
    return NULL;
  }

  return t;
} 


// frees this tensor
void free_tensor(tensor_t* t) {
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


void tensor_fill(tensor_t* t) {
  for (int i = 0; i < t->total_size; i++) {
    t->data[i] = ((float*)t->data)[i];
  }

}

int
main (int argc, char* argv[]) {
  

  int shape[] = {2, 3};
  tensor_t* t = create_tensor(2, shape);
  

  printf("Created tensor at %p\n", (void*)t);
  printf("Dimensions: %d\n", t->ndim);
  
  printf("Shape: [%d, %d]\n", t->shape[0], t->shape[1]);
  
  tensor_fill(t);
  tensor_print(t);
  // tests: free tensor
  free_tensor(t);
  t = NULL;
  printf("Tensor freed\n");
  printf("Tensor location after freeing: %p\n", (void*)t);
  return 0;
}





