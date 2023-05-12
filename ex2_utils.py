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
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    return


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


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
