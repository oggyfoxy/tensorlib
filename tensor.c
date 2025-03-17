

#include <stdio.h>
#include <string.h>
#include "tensor.h"



tensor_t* create_tensor(int ndim, int* shape) {
  tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t)); // allocate memory for a tensor on the heap
   
} 

tensor_t* free_tensor(tensor_t* t) {
  if (t == NULL) return;
  if (t->data != NULL) free(t->data);
  free(t);
}

int
main (int argc, char* argv[]) {
  

  // TODO: have tensors be printed here
  printf("Tensor\n", create_tensor(3, 2, INT8));
  printf("Hello World\n");
  return 0;
}
