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
LAPLACIAN_MATRIX = np.array([0 ,1 , 0],
                            [1 ,-4, 1],
                            [0 , 1, 0])


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
    
    # For option " �border Type� = cv2.BORDER_REPLICATE ", extra padding:
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

    # Kernel size must be odd and positive
    if ((k_size % 2) == 0 || k_size < 0)
        print "The kernel dimension must be odd and positive"
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
    if ((k_size % 2) == 0 || k_size < 0)
        print "The kernel dimension must be odd and positive"
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
    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    Implementation which based on practitioner guidance.
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    # Verify there is a valid input
    if in_image.ndim != 2:
        print "ERROR: The input image must be a 2 dimensions image"
        return in_image, in_image
    
    row, col = in_image.shape
    half_kernel_size  = math.floor(k_size/2)

    # init
    filtered_image = np.empty([row, col])
    extra_pad_img = cv2.copyMakeBorder(in_image, half_kernel_size, half_kernel_size,
                             half_kernel_size, half_kernel_size, borderType=cv2.BORDER_REPLICATE)

    pad_row, pad_col = extra_pad_img.shape
    for x in range(half_kernel_size, pad_row - half_kernel_size):
        for y in range(half_kernel_size , pad_col - half_kernel_size):
            # Take pixel to flter
            pivot_v = extra_pad_img[x, y]
            # Take its neighborhood according to kernel size
            neighborhood = extra_pad_img[x - half_kernel_size : x + half_kernel_size  + 1,
                                          y - half_kernel_size : y + half_kernel_size  + 1]

            # The formula: newValue = (color_diff_factor * space_diff_factor * oldValue).sum()
            #                        / (color_diff_factor * space_diff_factor).sum()
            color_diff = pivot_v - neighborhood
            color_diff_factor = np.exp(-np.power(color_diff, 2) / (2 * sigma_color))

            gaus_kernel = cv2.getGaussianKernel(k_size, sigma_space)
            space_diff_factor = gaus_kernel.dot(gaus_kernel.T)

            bilateral_fact = space_diff_factor  * color_diff_factor
            filtered_image[x - half_kernel_size, y - half_kernel_size] = 
                                (bilateral_fact * neighborhood).sum() / bilateral_fact.sum()

    return cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space), filtered_image