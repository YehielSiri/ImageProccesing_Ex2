"""
        '########:'##::::'##::'########:::
         ##.....::. ##::'##:::::.....##:::
         ##::::::::. ##'##:::::::::::##:::
         ######:::::. ###::::: ########:::
         ##...:::::: ## ##:::: ##.....::::
         ##:::::::: ##:. ##::: ##:::::::::
         ########: ##:::. ##:: ########:::
        ........::..:::::..:::.........:::
"""


import math
import numpy as np
import cv2


DERIVATIVE_KERNEL = np.array([[1, 0, -1]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 204155311

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    array_len, kernel_len = in_signal.size, k_size.size
    
    # Padding with zeroes:
    pad_array = np.pad(in_signal, (kernel_len - 1,))

    # Flip the kernel array:
    flip_kernel = np.flip(k_size)

    # Convolute the array:
    return np.array([np.dot(pad_array[i:i + kernel_len], flip_kernel) for i in range (array_len + kernel_len - 1)])


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image

def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()
			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k
	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")
	# return the output image
	return output
    """
    # Set the convolution measurements according the kernel dimension:
    if ((kernel.ndim % 2) == 0)
        print "The kernel dimension must be odd"
        return in_image
    elif (kernel.ndim == 1):
        convHeight = 1
        height_pad = 0
        convWidth = kernel.shape[0]
        width_pad = int((convWidth - 1) / 2)    # Kernel must be odd
        kernel = kernel[-1::-1]
    else:
        convHeight = kernel.shape[0]
        height_pad = int((convHeight - 1) / 2)  # Kernel must be odd
        convWidth = kernel.shape[1]
        width_pad = int((convWidth - 1) / 2)    # Kernel must be odd
        kernel = kernel[-1::-1, -1::-1]
    
    # For option " ’border Type’ = cv2.BORDER_REPLICATE ", extra padding:
    """
        if c == 27:
            break
        elif c == 99: # 99 = ord('c')
            borderType = cv.BORDER_CONSTANT
        elif c == 114: # 114 = ord('r')
            borderType = cv.BORDER_REPLICATE
    """
    padded_image = cv2.copyMakeBorder(in_image, height_pad, height_pad, width_pad, width_pad, borderType = cv2.BORDER_REPLICATE)

    result_image = np.zeros(in_image.shape)
    for i in range(result_image.shape[0]):
        for j in range(result_image.shape[1]):
            sub_image = padded_image[i: i + convHeight, j: j + convWidth]
            result_image[i, j] = (sub_image * kernel).sum()

    return result_image


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """
    #kernel = np.array([[1, 0, -1]])
    #kernel_transpose = kernel.T

    x_der_image = conv2D(in_image, DERIVATIVE_KERNEL)
    y_der_image = conv2D(in_image, DERIVATIVE_KERNEL.T)

    # Compute magnitude matrix  =>  sqrt( Xi^2 + Yi^2 )
    magnitude = np.sqrt(np.square(x_der_image) + np.square(y_der_image))
    # Compute direction matrix  =>  tan^-1(Yi/Xi)  =>  arctan2(Yi/Xi)
    directions = np.arctan2(y_der_image, x_der_image)

    return magnitude, directions


def gaussianKernel(size: int, sigma: float = 1) -> np.ndarray:
    """
    Compute a Gaussian kernel in a required size => sigma i,j = 1
    :param size: Kernel size
    :param : Kernel SIGMA  -  SIGMA kernel i,j = 1
    :return: The Gaussian kernel
    """
    center = size // 2
    kernel = np.zeros((size, size))

    for i in range (size):
        for j in range (size):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # Test & show my gaussian kernel function:
    plt.imshow(gaussianKernel(5))

    return conv2D(in_image, gaussianKernel(k_size))


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
