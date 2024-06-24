#!/usr/bin/env python

try:
    import torch.cuda as tc
except Exception as e:
    print(f"Caught Exception during library loading: {e}")
    raise e

# check for the presence of a gpu
def get_accelerator_device():
    # assume no gpu is present
    accelerator = "cpu"

    # test the presence of a GPU...
    print(f"Checking for the availability of a GPU...")
    if tc.is_available():
        device_name = tc.get_device_name()
        device_capabilities = tc.get_device_capability()
        device_available_mem, device_total_mem = [x / 1024**3 for x in tc.mem_get_info()]
        print(f"A GPU is available! [{device_name} - {device_capabilities} - {device_available_mem}/{device_total_mem} GB VRAM]")
        accelerator = "cuda"

    # return any accelerator found
    return accelerator
