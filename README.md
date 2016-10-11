# ImageSytleTransfer
An implementation for the paper http://arxiv.org/abs/1508.06576. Both Caffe version and Tensorflow version are provided.

Example code can be run by 

```python CaffeImageSytleTransfer.py --style <style image> --content <content image> --out_file <output file name>```

```python TensorImageStyleTransfer.py --style <style image> -- content <content image> --out_file <output file name>```

More details can be found in ```parseArg()``` method in each converter.

An example output is:
Style and content image: 
<p align = "center">
<img src="https://github.com/consciousgaze/ImageSytleTransfer/blob/master/starry_night.jpg" width="45%"/>
<img src="https://github.com/consciousgaze/ImageSytleTransfer/blob/master/full_moon.jpg" width="45%"/>
</p>

Caffe output image:
<p align = "center">
<img src="https://github.com/consciousgaze/ImageSytleTransfer/blob/master/full_moon_output.jpg" width="65%"/>
</p>


Tensorflow output image:
<p align = "center">
<img src="https://github.com/consciousgaze/ImageSytleTransfer/blob/master/full_moon_tensor_output.jpg" width="65%"/>
</p>
