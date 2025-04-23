## Set up logs (2025.04.22)
* `GraphDreamer` requires CUDA 11.7 for compatibility with `tiny-cuda-nn` package. Thus in order to run in on `capybara` (Ubuntu22.04, default cuda12), I installed [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) and [cudnn](https://developer.nvidia.com/rdp/cudnn-archive) ("Download cuDNN v8.9.7 for CUDA 11.x"). Since the default cuda is `cuda12`, I put environment variable setting in `setup_cuda.sh`. We need to run `. scripts/setup_cuda.sh` or `source scripts/setup_cuda.sh` before running any example script, e.g. `bash scripts/blue_jay.sh`

### Bugs and Solutions
* Bug: "The detected CUDA version (12.2) mismatches the version that was used to compile PyTorch (11.7)." <br>
Reason: CUDA version mismatch <br>
Soln: download and install CUDA 11.7
```
export CUDA_HOME=/usr/local/cuda-11.7
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
Then rebuild tiny-cuda-nn
* Bug: "ImportError: libcudnn.so.8: cannot open shared object file: No such file or directory" <br>
Reason: missing cuDNN <br>
Soln: Manually downloaded cuDNN 8.x for CUDA 11.7.
* Bug: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.5." <br>
Soln: downgrade numpy  `pip install numpy==1.24.4`
* Bug: "AttributeError: module 'torch' has no attribute 'compiler'" <br>
Soln: specify compatible version for packages in requirements.txt
```
transformers==4.30.2
diffusers == 0.21.4
huggingface_hub==0.30.2
```
* Bug: "ImportError: cannot import name 'driver' from 'triton.runtime'" <br>
Soln: Uninstalled triton and downgraded bitsandbytes
```
pip uninstall bitsandbytes triton
pip install bitsandbytes==0.39.0
```

* Bug: "cannot import name 'cached_download' from 'huggingface_hub'" <br>
Soln: In "env/lib/python3.8/site-packages/diffusers/utils/dynamic_modules_utils.py", remove `cached_download` from "import" and replace `cached_download` with `hf_hub_download`
