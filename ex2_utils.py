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
import matplotlib.pyplot as plt


DERIVATIVE_KERNEL = np.array([[1, 0, -1]])
LAPLACIAN_MATRIX = np.array([[0 ,1 , 0],
                            [1 ,-4, 1],
                            [0 , 1, 0]])
HOUGH_THRESHOLD = 0.7


def myID() -> int:
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
    if (kernel.ndim == 1):
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
    
    # For option " �border Type� = cv2.BORDER_REPLICATE ", extra padding:
    padded_image = cv2.copyMakeBorder(in_image, height_pad, height_pad, width_pad, width_pad, borderType = cv2.BORDER_REPLICATE)

    row, col = (0, 0)
    try:
        row, col = in_image.shape
    except AttributeError:
        print('AttributeError: value is', in_image)

    result_image = np.zeros((row, col), dtype="float32")
            
    for i in range(row):
        for j in range(col):
            sub_image = padded_image[i: i + convHeight, j: j + convWidth]
            result_image[i, j] = (sub_image * kernel).sum()

    return result_image


def convDerivative(in_image: np.ndarray) -> (tuple[np.ndarray, np.ndarray]):
    """
    Calculate gradient of an image
    :param in_image: Grayscale image
    :return: (directions, magnitude)
    """

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
    # Kernel size must be odd and positive
    if (k_size % 2) == 0 or k_size < 0:
        print ("The kernel dimension must be odd and positive")
        return in_image

    return conv2D(in_image, gaussianKernel(k_size))


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # Kernel size must be odd and positive
    if ((k_size % 2) == 0 or k_size < 0):
        print ("The kernel dimension must be odd and positive")
        return in_image
    
    # Build a gaussian kernel. cv2 creata an 1D one!
    kernel1D = cv2.getGaussianKernel(k_size, -1)
    kernel2D = kernel1D @ kernel1D.transpose()

    return cv2.filter2D(in_image, -1, kernel2D, borderType = cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def findZCPatterns(img: np.ndarray) -> np.ndarray:
    """
    Find zero-crossing patterns in a 'Laplacian of Gaussian'
    filtered image
    :param img: LoG filtered image
    :return: An edge matrix
    """
    edge_img = np.zeros(img.shape)

    row, col = img.shape
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if img[i, j] == 0:
                if (   (   (img[i - 1, j] > 0 and img[i + 1, j] < 0)
                        or (img[i - 1, j] < 0 and img[i + 1, j] > 0)   )
                    or (   (img[i, j - 1] > 0 and img[i, j + 1] < 0)
                        or (img[i, j - 1] < 0 and img[i, j + 1] > 0)   )   ):
                    edge_img[i, j] = 1
            elif img[i, j] > 0:
                if (   (img[i - 1, j] < 0 or img[i + 1, j] < 0)
                    or (img[i, j - 1] < 0 or img[i, j + 1] < 0)   ):
                    edge_img[i, j] = 1
            # img[i, j] < 0
            else:
                if img[i - 1, j] > 0:
                    edge_img[i - 1, j] = 1
                elif img[i + 1, j] > 0:
                    edge_img[i + 1, j] = 1
                elif img[i, j - 1] > 0:
                    edge_img[i, j - 1] = 1
                elif img[i, j + 1] > 0:
                    edge_img[i, j + 1] = 1
    return edge_img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    # Preper the LoG kernel (a single kernel for two kernels; Gaussian with Laplacian)
    LoG_filter = conv2D(gaussianKernel(3), LAPLACIAN_MATRIX)

    # Smooth and apply Laplacian in a single convolution
    log_filtered_img = conv2D(img, LoG_filter)

    # Find zero-crossing patterns
    return findZCPatterns(log_filtered_img)


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
    # init
    circles = list()
    
    # Calculate derivatives on the both axis, x and y, using cv2.Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, HOUGH_THRESHOLD)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, HOUGH_THRESHOLD)

    # Directions
    direction = np.radians(np.arctan2(sobel_x, sobel_y) * 180 / np.pi)
    accumulator = np.zeros((len(img), len(img[0]), max_radius+1))

    # Find image's edges using cv2.Canny
    edges = cv2.Canny((img * 255).astype(np.uint8), 0.1, 0.45)

    height = len(edges)
    width = len(edges[0])
    for x in range(0, height):
        for y in range(0, width):
            if edges[x][y] == 255:
                for radius in range(min_radius, max_radius + 1):
                    angle = direction[x, y] - np.pi / 2
                    x1, x2 = np.int32(x - radius * np.cos(angle)), np.int32(x + radius * np.cos(angle))
                    y1, y2 = np.int32(y + radius * np.sin(angle)), np.int32(y - radius * np.sin(angle))
                    if 0 < x1 < len(accumulator) and 0 < y1 < len(accumulator[0]):
                        accumulator[x1, y1, radius] += 1
                    if 0 < x2 < len(accumulator) and 0 < y2 < len(accumulator[0]):
                        accumulator[x2, y2, radius] += 1

    thresh = np.multiply(np.max(accumulator), 1/2)

    x, y, radius = np.where(accumulator >= thresh)
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0 and radius[i] == 0:
            continue
        circles.append((y[i], x[i], radius[i]))
    
    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        tuple[np.ndarray, np.ndarray]):
    """
    Implementation which based on practitioner guidance.
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    # init
    half_k = int(k_size / 2)
    new_img = np.pad(in_image, ((half_k, half_k), (half_k, half_k)), mode='edge').astype('float32')
    result = [[(__bilateral_pixle(new_img, i + half_k, j + half_k, k_size, sigma_color, sigma_space))
               for j in range(len(in_image[0]))] for i in range(len(in_image))]
    result = np.array(np.rint(result)).astype('int')
    opencv_result = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space, cv2.BORDER_REPLICATE)
    return opencv_result, result


def __bilateral_pixle(in_image: np.ndarray, y, x, k_size: int, sigma_color: float, sigma_space: float):
    # The formula: newValue = (color_diff_factor * space_diff_factor * oldValue).sum()
    #                        / (color_diff_factor * space_diff_factor).sum()
    img = in_image
    mid_kernel = int(k_size / 2)
    pivot = img[y, x]  # the color of the target
    neighbor_hood = img[y - mid_kernel:y + mid_kernel + 1, x - mid_kernel:x + mid_kernel + 1]
    diff = pivot - neighbor_hood
    diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color ** 2))
    distance_gau = cv2.getGaussianKernel(k_size, sigma_space)
    distance_gau = distance_gau.dot(distance_gau.T)
    combo = distance_gau * diff_gau
    return (combo * neighbor_hood).sum() / combo.sum()
