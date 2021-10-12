import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# Task 1a
def convert_to_gray_scale(image):
    """
    This function takes the read image as input and displays the image as  it is and also converts it to gray scale

    :param: color image
    :return: gray scale version of the image
    """
    _grey_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original image", image/255)
    # cv2.imshow("grey_scale_image", _grey_scale_image/255)
    # cv2.waitKey()
    return _grey_scale_image


# Task 1b
def gaussian_kernel(_sigma_values, image):
    """
    This function creates gaussian kernels for each of the sigma values covering -/+3 sigma (total of 6 sigma) range
    and applies it to the image, to create 12 smoothed images

    :param _sigma_values: list of 12 values
    :param image: gray scale image
    :return: list of 12 gaussian kernels and list of 12 Gaussian smoothed images
    """

    _kernels = list()
    _filtered_images = list()

    for sigma in _sigma_values:
        x, y = np.meshgrid(np.arange(-3 * sigma, 3 * sigma),
                           np.arange(-3 * sigma, 3 * sigma))
        gaussian_x_y = (1 / (2 * np.pi * sigma**2)) * (np.exp(-(x**2 + y**2)/(2 * (sigma**2))))
        _kernels.append(gaussian_x_y)
        # print("shape of kernel: ", gaussian_x_y.shape)
        # print("for sigma: {} max value of kernel is: {}".format(sigma, np.max(gaussian_x_y)))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x, y, gaussian_x_y)
        # plt.title("Gaussian Kernel sigma =  %s" % sigma)
        # plt.figure()
        # plt.imshow(gaussian_x_y)
        # plt.title("Gaussian Kernel sigma = %s"% sigma)
        # plt.show()
        filtered_image = cv2.filter2D(image, -1, gaussian_x_y)
        _filtered_images.append(filtered_image)
        # cv2.imshow("filtered image with sigma: {}".format(sigma), filtered_image/255)
        # plt.show()
        # cv2.waitKey()

    return _kernels, _filtered_images


# Task 1b
def gaussian_smoothing(image):
    """
    This function takes creates sigma values in the range 2**(k/2). In other words these are the octaves for SIFT

    :param: gray scale image

    :return: list sigma values using the formula 2**(k/2) where k ranges from 0 to 11 (both inclusive), gaussian
             kernels generated using the sigma values and the smoothed images obtained after applying the gaussian
             kernel
    """
    _sigma_values = []
    for k in range(12):
        _sigma_values.append(2**(k/2))

    _kernels, _filtered_images = gaussian_kernel(_sigma_values, image)
    return _sigma_values, _kernels, _filtered_images


# Task 2a
def compute_difference_of_gaussians(_sigma_values, _kernels, _filtered_images):
    """
    computes gaussian difference between the gaussian smoothed images:image sigma2 - image sigma1, where sigma2 > sigma1
    :param _sigma_values: list of sigma values
    :param _kernels: list of 12 kernels for each sigma in _sigma_values list
    :param _filtered_images: list of 12 gaussian smoothed images
    :return: Dictionary with min sigma of the two smoothed images as key and the DoG image. Length of this dict is 12
    """
    gaussian_difference_images_dict = {}
    total_kernels = len(_kernels)
    print("\nComputing Difference of Gaussians")
    print("Sigma values: ", _sigma_values)
    print("number of kernels: ", len(_kernels))
    print("number of gaussian images: ", len(_filtered_images))

    print("Calculation DoG images: ")
    for i in range(total_kernels - 1):
        difference_of_gaussian = _filtered_images[i + 1] - _filtered_images[i]
        gaussian_difference_images_dict[min(_sigma_values[i], _sigma_values[i+1])] = difference_of_gaussian
        # plt.imshow(difference_of_gaussian, cmap="gray")
        # plt.show()

    print("Number of DoG images: ", len(gaussian_difference_images_dict))
    return gaussian_difference_images_dict


# Task 2b
def identify_key_points(gaussian_difference_images_dict):
    """
    This function identifies the key points or feature points or point of interest in the image making use of DoG images
    :param gaussian_difference_images_dict: dictionary returned from the function "compute_difference_of_gaussians"
    :return: key points which is a list of dictionaries, each dictionary is an identified key point storing its x and y
             coordinates along with the sigma/scale value at which the key point is found
    """
    threshold = 10
    _key_points = []
    _sigma_values = list(gaussian_difference_images_dict.keys())
    dog_images = list(gaussian_difference_images_dict.values())

    for i in range(len(dog_images)):
        current_image = dog_images[i]

        # Below lines take care of the boundary conditions of first DoG and last DoG to compare only next and previous scale respectively
        next_image = np.zeros(current_image.shape) if i == len(dog_images) - 1 else dog_images[i + 1]
        previous_image = np.zeros(current_image.shape) if i == 0 else dog_images[i - 1]

        for x in range(1, current_image.shape[0] - 1):
            for y in range(1, current_image.shape[1] - 1):
                current_pixel = current_image[x][y]

                if current_pixel > threshold:  # Thresholding the pixel values
                    '''We have to compare each pixel with its 26 neighbours (8 in the same scale, 9 + 9 in the adjacent
                       scale space, if the current pixel is maximum of all the neighbours. This is called as non-maxima
                       suppression'''
                    values_to_compare_with = [current_image[x, y+1], current_image[x, y-1], current_image[x-1, y],
                                              current_image[x+1, y], current_image[x+1, y+1], current_image[x-1, y+1],
                                              current_image[x+1, y-1], current_image[x-1, y-1], previous_image[x, y],
                                              previous_image[x, y + 1], previous_image[x, y - 1], previous_image[x - 1, y],
                                              previous_image[x + 1, y], previous_image[x + 1, y + 1], previous_image[x - 1, y + 1],
                                              previous_image[x + 1, y - 1], previous_image[x - 1, y - 1], next_image[x, y],
                                              next_image[x, y + 1], next_image[x, y - 1], next_image[x - 1, y], next_image[x + 1, y],
                                              next_image[x + 1, y + 1], next_image[x - 1, y + 1], next_image[x + 1, y - 1],
                                              next_image[x - 1, y - 1]]

                    # Non Maxima suppression
                    if current_pixel > max(values_to_compare_with):
                        _key_points.append({'x': x,
                                            'y': y,
                                            'sigma': _sigma_values[i]})
    return _key_points


# Task 2c
def visualise_key_points(_key_points, _original_image):
    for point in _key_points:
        cv2.circle(_original_image, (point['y'], point['x']), int(3*point['sigma']), (255, 0, 0), thickness=1)
    cv2.imshow("result", _original_image / 255)
    cv2.waitKey()


# Task 3a
def derivatives_scale_space_images(_filtered_images, _sigma_values):
    """
    This function calculates the gradients along x and y axis which determines vertical and horizontal
    edges respectively
    :param _filtered_images: list of 12 Gaussian smoothed images
    :param _sigma_values: list of 12 sigma values obtained from gaussian_smoothing function
    :return: list of dictionaries, where each dictionary contains the x and y derivative images along with the scale
             value (sigma) of the smoothed image
    """
    x = np.array([[1, 0, -1]])
    y = x.T
    _derivative_images = []
    for i, image in enumerate(_filtered_images):
        dgx = cv2.filter2D(image, -1, x)
        dgy = cv2.filter2D(image, -1, y)
        _derivative_images.append({'sigma': _sigma_values[i],
                                   'dgx': dgx,
                                   'dgy': dgy})

        # cv2.imshow("dgx for sigma : {}".format(_sigma_values[i]), dgx)
        # cv2.imshow("dgy for sigma:  {}".format(_sigma_values[i]), dgy)
        # cv2.waitKey()
    return _derivative_images


# Task 3bc
def key_point_orientation(image_derivatives, _key_points):
    """
    Compute gradient lengths, gradient directions and weighted contribution of each gradient for each orientation (Among
    36 bins ranging from -180 to +170 with 10 degree bins each
    :param image_derivatives: list of x and y derivative image dictionaries obtained from function
                              derivatives_scale_space_images
    :param _key_points: list of key point dictionaries obtained from function identify_key_points
    :return: list of key point orientation dictionaries where each dictionary contains x and y coordinates of the key
             point, scale (sigma) at which the key point was identified and theta (orientation (angle) of the key point)
    """
    _key_point_orientation = []
    tot_mqr = []

    nx, ny = (7, 7)
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)

    # Create histogram bins (list) with values between -180 degree and +180 degree with a step of 10 degrees
    hist_bins = np.arange(-180, 180, 10)
    for key_point in _key_points:
        q, r = np.meshgrid((3 / 2) * x * key_point['sigma'], (3 / 2) * y * key_point['sigma'])

        # Get the relative coordinates of 7X7 grid with respect to key point coordinates and rounding off to get the
        # Nearest neighbour interpolation
        qplusx = np.round(q + key_point['x']).astype(int)
        rplusy = np.round(r + key_point['y']).astype(int)

        try:
            # Get the Gaussian derivative images (dx and dy) with same scale as of the current key point
            required_dict = list(filter(lambda derivatives: derivatives['sigma'] == key_point['sigma'],
                                        image_derivatives))[0]

            # Gradient lengths
            mqr = np.sqrt(np.square(required_dict['dgy'][qplusx, rplusy]) +
                          np.square(required_dict['dgx'][qplusx, rplusy]))

            # Gradient directions. arctan2 will return the angle in radians, so we are converting it to degrees
            # arctan2 function will return the radian angle in the range -pi to +pi
            thetaqr_rad = np.arctan2(required_dict['dgy'][qplusx, rplusy], required_dict['dgx'][qplusx, rplusy])
            thetaqr_deg = np.rad2deg(thetaqr_rad)
            tot_mqr.append(mqr)

            # Gaussian Weighting function
            exp_num = q ** 2 + r ** 2
            exp_den = (9 * (key_point['sigma'] ** 2)) / 2
            numerator = np.exp(-exp_num / exp_den)
            denominator = (9 * np.pi * key_point['sigma'] ** 2) / 2
            wqr = numerator / denominator

            # Weighted gradient lengths
            weighted_gradient_lengths = wqr * mqr

            # Return the indices for each of the 49 orientation angles with respect to its histogram bin
            hist_indices = np.digitize(thetaqr_deg.flatten(), hist_bins) - 1

            # Sum up the respective bin values
            histograms = np.bincount(hist_indices.flatten(), weighted_gradient_lengths.flatten(), minlength=36)

            # Take the orientation as the value of the max value in the histogram bin
            orientation_angle = hist_bins[np.argmax(histograms)]
            _key_point_orientation.append({'x': key_point['x'],
                                           'y': key_point['y'],
                                           'sigma': key_point['sigma'],
                                           'theta': float(orientation_angle)})

        except Exception as e:
            continue

    print("\nlen of key_point_orientation list: ", len(_key_point_orientation))
    return _key_point_orientation


# Task 3d
def visualise_key_point_orientations(image, _key_point_orientations):
    """
    Draws the line from center of the circle (centered at key points) stretching till the circumference of the circle
    as per the orientation
    :param image: original color image with drawn key points on it
    :param _key_point_orientations: list of dictionaries obtained from the function key_point_orientation
    :return:
    """

    for kpo in _key_point_orientations:
        cv2.circle(image, (kpo['y'], kpo['x']), int(3 * kpo['sigma']), (255, 0, 0), thickness=1)
        orient = (int(np.round(np.cos(kpo['theta']) * int(3 * kpo['sigma']))), int(np.round(np.sin(kpo['theta']) * int(3 * kpo['sigma']))))
        cv2.line(image, (kpo['y'], kpo['x']), (kpo['y'] + orient[0], kpo['x'] + orient[1]), (0, 255, 0), 2)
    cv2.imshow("key_point_orientations", image/255)
    cv2.waitKey()


# Task 4abc
def identify_feature_descriptors(image_derivatives, _key_point_orientations):
    """
    Identifies the 128 vector feature descriptor for each of the key_points whose orientations are obtained from the
    function key_point_orientation
    :param image_derivatives: x and y derivative images obtained from function derivatives_scale_space_images
    :param _key_point_orientations: list of dictionaries of key points with orientation obtained from function
                                    key_point_orientation
    :return: list of key point feature descriptors dictionaries for each key point, consisting of x and y coordinates,
             scale(sigma) at which the key point is identified and the 128 length vector (feature descriptor)
    """
    nx, ny = (4, 4)
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)

    # Create histogram bins (list) with values between 0 degree and 360 degree with a step of 45 degrees
    hist_bins = np.arange(0, 360, 45)
    _key_points_feature_descriptor = []
    for key_point in _key_point_orientations:
        # Get the Gaussian derivative images (dx and dy) with same scale as of the current key point
        required_dict = list(filter(lambda derivatives: derivatives['sigma'] == key_point['sigma'],
                                    image_derivatives))[0]
        key_point_hist = []
        for i in range(-2, 2, 1):
            for j in range(4):
                s, t = np.meshgrid((3 / 16) * x * (4*i + j + 0.5) * key_point['sigma'],
                                   (3 / 16) * y * (4*i + j + 0.5) * key_point['sigma'])
                # Get the relative coordinates of 4X4 subgrid with respect to key point coordinates and
                # rounding off to get the Nearest neighbour interpolation
                splusx = np.round(s + key_point['x']).astype(int)
                tplusy = np.round(t + key_point['y']).astype(int)

                try:
                    # Gaussian weighting function
                    exp_num = s ** 2 + t ** 2
                    exp_den = (81 * (key_point['sigma'] ** 2)) / 2
                    numerator = np.exp(-exp_num / exp_den)
                    denominator = (81 * np.pi * key_point['sigma'] ** 2) / 2
                    wst = numerator / denominator

                    # gradient lengths
                    mst = np.sqrt(np.square(required_dict['dgy'][splusx, tplusy]) +
                                  np.square(required_dict['dgx'][splusx, tplusy]))

                    # Gradient directions. arctan2 will return the angle in radians, so we are converting it to degrees
                    # arctan2 function will return the radian angle in the range -pi to +pi
                    thetast_rad = np.arctan2(required_dict['dgy'][splusx, tplusy], required_dict['dgx'][splusx, tplusy])
                    thetast_deg = (np.rad2deg(thetast_rad) % 360) - (key_point['theta'] % 360)
                    '''We are doing %360 to do away with -ve angles and have them between 0 and 360'''

                    # Calculate weighted gradient lengths for the subgrid of size 4X4
                    weighted_gradient_lengths = wst * mst

                    # Return the indices for each of the 49 orientation angles with respect to its histogram bin
                    hist_indices = np.digitize(np.abs(thetast_deg).flatten(), hist_bins) - 1

                    # Sum up the respective bin values
                    histograms = np.bincount(hist_indices.flatten(), weighted_gradient_lengths.flatten(), minlength=8)

                    # Collect accumulated histogram in a list to be concatenated later as a 128 dimensional vector
                    key_point_hist.append(histograms)

                except Exception as e:
                    print(e)
                    continue

        # Concatenating the 16-histogram 8 vectors resulting in a vector of length 128
        feature_descriptor = np.concatenate(key_point_hist)

        # Normalizing the feature descriptors for each key point
        normalized_feature_descriptor = feature_descriptor / (np.sqrt(np.dot(feature_descriptor.T, feature_descriptor)))

        # Capping the normalized feature descriptor values to 0.2
        capped_feature_descriptor = np.clip(normalized_feature_descriptor, a_min=np.min(normalized_feature_descriptor),
                                            a_max=0.2)
        # storing the key points along with feature descriptors for each key point
        _key_points_feature_descriptor.append({'x': key_point['x'],
                                               'y': key_point['y'],
                                               'sigma': key_point['sigma'],
                                               'feature_descriptor': capped_feature_descriptor})

        # print("Histogram of key point gradient lengths: ")
        # print(key_point_hist)
        #
        # print("Feature descriptors obtained on concatenating the histogram of shape 16X8 : ")
        # print(feature_descriptor)
        # print("Length of feature descriptors: ", len(feature_descriptor))
        #
        # print("\nNormalized feature descriptors:")
        # print(normalized_feature_descriptor)
        # print("Length of normalized feature descriptors: ", len(normalized_feature_descriptor))
        #
        # print("Capped_feature_descriptors: ")
        # print(capped_feature_descriptor)
        # print("Length of capped feature descriptors: ", len(capped_feature_descriptor))
        # exit()
    print("Len of key points feature descriptors: ", len(_key_points_feature_descriptor))
    return _key_points_feature_descriptor


if __name__ == "__main__":
    sift_start_time = time.process_time()
    original_image = cv2.imread("../Assignment_MV_1_image.png")  # Load the image
    original_image = original_image.astype(np.float32)
    print("Original image shape: ", original_image.shape)
    print("Original image data type: ", original_image.dtype)

    grey_scale_image = convert_to_gray_scale(original_image)
    print("\nGrey scale image shape: ", grey_scale_image.shape)
    print("Grey scale image data type: ", grey_scale_image.dtype)

    print("\nmax of original image: ", np.max(original_image))
    print("\nmax of grey scale image: ", np.max(grey_scale_image))

    start_time = time.process_time()
    sigma_values, kernels, filtered_images = gaussian_smoothing(grey_scale_image)
    end_time = time.process_time()
    print("Time taken to smooth image using gaussian kernels: ", end_time-start_time)

    start_time = time.process_time()
    dog_dict = compute_difference_of_gaussians(sigma_values, kernels, filtered_images)
    end_time = time.process_time()
    print("\nTime taken to compute DoGs: ", end_time - start_time)

    start_time = time.process_time()
    key_points = identify_key_points(dog_dict)
    end_time = time.process_time()
    print("\nTime taken to identify key points: ", end_time - start_time)
    print("\ntotal key points: ", len(key_points))

    start_time = time.process_time()
    visualise_key_points(key_points, original_image)
    end_time = time.process_time()
    print("\nTime taken to visualise the key points: ", end_time - start_time)

    start_time = time.process_time()
    derivative_images = derivatives_scale_space_images(filtered_images, sigma_values)
    end_time = time.process_time()
    print("\nTime taken to compute x and y derivatives of scale space images: ", end_time - start_time)

    start_time = time.process_time()
    key_point_orientations = key_point_orientation(derivative_images, key_points)
    end_time = time.process_time()
    print("\nTime taken to find out key point orientations: ", end_time - start_time)

    visualise_key_point_orientations(original_image, key_point_orientations)

    start_time = time.process_time()
    key_points_feature_descriptor = identify_feature_descriptors(derivative_images, key_point_orientations)
    end_time = time.process_time()
    sift_end_time = time.process_time()
    print("\nTime taken to identify feature descriptors for each key point: ", end_time - start_time)

    print("\nTime taken to complete the sift algo: ", sift_end_time - sift_start_time)
