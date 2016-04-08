<div style="background:#abcfbb">

# wincuda

Demo of Julia algorithm using cuda (if supported by the GPU), to compute a 1024 x 1024 image and compares performance with a CPU version of the algorithm.

Cuda version utilizes 64 kernels by 16 threads each.

###Features

Menu item `Settings` selects the version of the algorithm.<br>
A status bar at the bottom of the window displays the version and the computation time. For comparison reasons, the image is drawn in different colors, <span style="color:white">white</span> for GPU,
<span style="color:green">green</span> for CPU.<br>

Menu item `About` displays basic information about the device.

