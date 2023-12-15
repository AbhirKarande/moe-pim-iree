---

# IREE-Runtime HAL driver Interface Implementation

This directory include IREE-runtimes internal backend driver implementation

## PIM backend integrate to IREE-Runtime 

We implement example PIM backend to modify vulkan backend driver implementation.
Some remnants of Vulkan-related code still exist in several places, but they do not have any impact on the actual execution of the PiM backend. 
The list of our main PiM backend implementation is as follows:

```bash
vulkan/PIM_allocator.cc
vulkan/PIM_allocator.h
vulkan/PIM_buffer.cc
vulkan/PIM_buffer.h
vulkan/direct_command_buffer.cc
vulkan/direct_command_buffer.h
vulkan/PIM_device.cc
vulkan/PIM_device.h
vulkan/PIM_driver.cc
vulkan/PIM_driver.h
vulkan/registration/driver_module.cc
vulkan/registration/driver_module.h
```





