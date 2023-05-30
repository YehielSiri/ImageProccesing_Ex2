# Image Convolution and Edge Detection
Assignment number 2 in Computer Vision & Image Proccessing course. All tasks and functions were written in the ex2_utils.py file and they were all tested in the ex2_main.py file.

-----

Using Python and the OpenCV library, implementation of:
1. [ Convolution ](#convolution)
2. [ Image derivatives ](#image-derivatives)
3. [ Image Blurring ](#image-blurring)
4. [ Edge detection via LoG zero-crossing ](#edge-detection-via-log-zero-crossing)
5. [ Hough Circles ](#hough-circles)
6. [ Bilateral filter ](#bilateral-filter)

-----

<h2>Convolution</h2>

<div align="center">

| Comparing 2D Convolution Output to OpenCV |
| ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/conv2D_beach.png"/></p>  |
MSE: 0.00028858193628147563
Max Error: 0.4800071418285348
  
 </div>

-----

<h2>Image derivatives</h2>
Calculate gradient of an image. An image derivative - gradient - is defined as the change in the pixel value of an image.

<div align="center">
  
| Directions (angles) / Magnitude |
| ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/ImageDerivatives_beach_OriMag.png"/></p>  |
  
</div>

-----

<h2>Image Blurring</h2>
Image blurring is achieved by convolving the image with a low-pass filter kernel. It is useful for removing noise. It actually removes high frequency content (eg: noise, edges) from the image. So edges are blurred a little bit in this operation (there are also blurring techniques which don't blur the edges).
Here we are blurring an image using a Gaussian kernel, my implementation.

<div align="center">
  
| OpenCV blurred | My blurred |
| ------------- | ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/blurImage_beach.jpg"/></p>  |
|  | MSE:0.001328 |
  
</div>

-----

<h2>Edge detection via LoG zero-crossing</h2>
In edge detection, we find the boundaries or edges of objects in an image, by determining where the brightness of the image changes dramatically. Edge detection can be used to extract the structure of objects in an image. If we are interested in the number, size, shape, or relative location of objects in an image, edge detection allows us to focus on the parts of the image most helpful, while ignoring parts of the image that will not help us.
Detecting edges using "ZeroCrossingLOG" method - Zero-Crossing Detector Using the Laplacian of Gaussian (LoG) Filter

<div align="center">
  
| Comparing LoG zero-crossing Output to OpenCV |
| ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/edgeDetectionZeroCrossingLOG_boxMan.png"/></p>  |
  
</div>

-----

<h2>Hough Circles:</h2>
The circle Hough Transform (CHT) is a basic feature extraction technique used in digital image processing for detecting circles in imperfect images. The circle candidates are produced by “voting” in the Hough parameter space and then selecting local maxima in an accumulator matrix.
Find Circles in an image using a Hough Transform algorithm extension.

| Original | Output | Original | Output |
| ------------- | ------------- | ------------- | ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/input/coins.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/houghCircles_coins.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/input/kidsPlay.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/houghCircles_kidsPlay.png"/></p>  |
| | Time[Mine]: 43.060 sec [CV]: 0.051 sec | | Time[Mine]: 0.664 sec [CV]: 0.004 sec |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/input/pool_balls.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/houghCircles_pool_balls.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/input/ruler.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/houghCircles_ruler.png"/></p>  |
| | Time[Mine]: 1.766 sec [CV]: 0.005 sec | | [Mine]: 2.415 sec [CV]: 0.001 sec |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/input/tapet.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/houghCircles_tapet.png"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/input/beach_ball_org.png"/></p> | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/beach_ball.png"/></p>  |
| | [Mine]: 3.333 sec [CV]: 0.006 sec | | |

-----

<h2>Bilateral filter</h2>

<div align="center">
  
| Original | OpenCV | Mine |
| ------------- | ------------- | ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/input/boxMan.jpg"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/OpenCVBilateralFilter_boxMan.jpg"/></p>  | <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/myBilateralFilter_boxMan.jpg"/></p>  |
| | | MSE: 0.003615234375 Max Error: 4 |
  
</div>

-----
