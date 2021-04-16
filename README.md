# flownet2-pytorch-module
Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks.

Rearrange all code of nvidia flownet2-pytorch [repository](https://github.com/NVIDIA/flownet2-pytorch) for simplicity.

Let flownet be a module that can be plugged into any code easily.

## Compile and install
* Activate your python envs.

  Compile testing successfully on the env settings:
  ```shell
  linux centos-7,
  gcc 5.5,
  python 3.6,
  pytorch-1.6,
  cuda-10.2
  ```
  You may need to install some required packages(e.g. python base packages, pytorch) and configure 
  env(e.g. cuda version matches gcc version and pytorch version) correctly.
* Compile src code then install on the activated env with running the following bash cmd.
  ```shell
  sh tools/install.sh
  ```
## Usage

* Download FlowNet2 pretrained [weight](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view).
* Test the installation and compile with following cmd:
  ```shell
  python tools/flownet_test.py 
  ```
* You can plug it into your code easily.
## Related repository

* https://github.com/NVIDIA/flownet2-pytorch