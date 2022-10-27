# 🍁 Maple Diffusion Swift Package

Maple Diffusion runs Stable Diffusion models **locally** on macOS / iOS devices, in Swift, using the MPSGraph framework (not Python).

This is a Swift Package Manager version of [Maple Diffusion](https://github.com/madebyollin/maple-diffusion). It also adds a few combine publishers and async/await versions of some functions.   


# Install
Add `https://github.com/mortenjust/maple-diffusion` in the ["Swift Package Manager" tab in Xcode](https://developer.apple.com/documentation/xcode/adding_package_dependencies_to_your_app)

# Preparing the weights

## Option 1: Pre-converted Standard Stable Diffusion v1.4
By downloading this zip file, you accept the [creative license from StabilityAI](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE). [Download ZIP](https://drive.google.com/file/d/1fGPc7-1upu-b68jstdT1vF7uWICc6Vk8/view?usp=sharing). Please don't use this URL in your software. 

We'll get back to what to do with it in a second.  

<details><summary>## Option 2: Preparing your own `ckpt` file<summary>

1. Download a Stable Diffusion model checkpoint to a folder, e.g. `~/Downloads/sd` ([`sd-v1-4.ckpt`](https://huggingface.co/CompVis/stable-diffusion-v1-4), or some derivation thereof)

2. Setup & install Python with PyTorch, if you haven't already. 

```
# Grab the converter script
cd ~/Downloads/sd
curl https://raw.githubusercontent.com/mortenjust/maple-diffusion/main/Converter%20Script/maple-convert.py > maple-convert.py

# may need to install conda first https://github.com/conda-forge/miniforge#homebrew
conda deactivate
conda remove -n maple-diffusion --all
conda create -n maple-diffusion python=3.10
conda activate maple-diffusion
pip install torch typing_extensions numpy Pillow requests pytorch_lightning
./maple-convert.py ~/Downloads/sd-v1-4.ckpt
```
The script will create a new folder called `bins`. We'll get back to what to do with it in a second.

</details>


# 




# FAQ


## How fast is it? 

Maple Diffusion should be capable of generating a reasonable image in a minute or two on a recent iPhone (I get around ~2.3s / step on an iPhone 13 Pro).

To attain usable performance without tripping over iOS's 4GB memory limit, Maple Diffusion relies internally on FP16 (NHWC) tensors, operator fusion from MPSGraph, and a truly pitiable degree of swapping models to device storage.

On macOS, Maple Diffusion uses slightly more memory (~6GB), to reach <1s / step.

## What does it look like?
![](demonstration.jpg)
![](screenshot.jpg)


# Usage

To add Diffusion to your project





1. Download a Stable Diffusion model checkpoint ([`sd-v1-4.ckpt`](https://huggingface.co/CompVis/stable-diffusion-v1-4), or some derivation thereof)

2. Download this repo

   ```bash
   git clone https://github.com/madebyollin/maple-diffusion.git && cd maple-diffusion
   ```

3. Convert the model into a bunch of fp16 binary blobs. You might need to install PyTorch and stuff.

   ```bash
   ./maple-convert.py ~/Downloads/sd-v1-4.ckpt
   ```

4. Open, build, and run the `maple-diffusion` Xcode project. You might need to set up code signing and stuff
