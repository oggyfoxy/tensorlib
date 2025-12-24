```
minimalist tensor library in c that abstracts memory on ARM and x86.

will focus on optimizing different hardware: android devices (arm64), Pico 2W, 
and the ESP32-WROVER_E

TODO
current: modular tensor representation and primitive op abstractions
- dynamic memory pool, will make it static if issues occuring. [V] 
- tensor initialization [V]
- base interface: tensor_get, tensor_set, tensor_print, tensor_fill (pre-ops) [V]
- primitive ops:  unary / binary ops [in progress]
- lazy allocation
- zero-copy
- movement ops [x]
- AVX / NEON vectorization [x]
- alignment, complex ops [x]
- tiny mnist, other NN [x]
- quantized dtypes [x] 
- full sgemm kernel (c++17) [x]

will set up a guide and testing suite once more advancements are made

inspired by ggmml, tinygrad and https://github.com/GandalfTea/tensorlib <3
```
