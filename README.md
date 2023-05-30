# Image-Convolution-and-Edge-Detection
Assignment number 2 in Computer Vision & Image Proccessing course.

-----

Implementation of:
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
Calculate gradient of an image.

<div align="center">
  
| Directions (angles) / Magnitude |
| ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/ImageDerivatives_beach_OriMag.png"/></p>  |
  
</div>

-----

<h2>Image Blurring</h2>
Blurring an image using a Gaussian kernel, my implementation.

<div align="center">
  
| OpenCV blurred | My blurred |
| ------------- | ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/blurImage_beach.jpg"/></p>  |
|  | MSE:0.001328 |
  
</div>

-----

<h2>Edge detection via LoG zero-crossing</h2>
Detecting edges using "ZeroCrossingLOG" method

<div align="center">
  
| Comparing LoG zero-crossing Output to OpenCV |
| ------------- |
| <p align="center"><img src="https://github.com/YehielSiri/ImageProccesing_Ex2/tree/main/TestingScreenshots/edgeDetectionZeroCrossingLOG_boxMan.png"/></p>  |
  
</div>

-----

<h2>Hough Circles:</h2>
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
