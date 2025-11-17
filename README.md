```
minimalist tensor library in c that abstracts memory on embedded devices.

will focus on android devices (arm64), Pico 2W, ESP32-WROVER_E

TODO:
- dynamic memory pool, will make it static if issues occuring. [V] 
- tensor initialization [V]
- tensor_get, tensor_set, tensor_print, tensor_fill (before ops) [V]
- unary / binary ops [in progress)
- android support: arm neon intrinsics [X]
- alignment, advanced ops, ... 

will set up a guide and testing suite once more advancements are made

inspired by tinygrad and https://github.com/GandalfTea/tensorlib <3
```
