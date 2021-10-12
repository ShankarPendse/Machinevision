import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

random.seed(195877)


# Camera calibration
def task1():
    image_points = []       # stores the image coordinates of the identified points with subpixel accuracy
    object_points = []      # Reference of real world coordinates with respect to identified image points

    images = glob.glob("../../../images/*.png")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # real world references
    objp = np.zeros((5 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=criteria)
            image_points.append(corners2)
            object_points.append(objp)

            img = cv2.drawChessboardCorners(img, (7, 5), corners2, ret)
            cv2.imshow("img", img)
            cv2.waitKey()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    principal_point = (mtx[0, 2], mtx[1, 2])
    principal_length = mtx[0, 0]
    aspect_ratio = mtx[1, 1] / mtx[0, 0]

    print("\nCalibration Matrix: \n", mtx)
    print("\nPrincipal Length =  ", principal_length)
    print("\nPrincipal point = ", principal_point)
    print("\nAspect Ratio = ", aspect_ratio)

    # Track good features
    vid = cv2.VideoCapture("../../../Assignment_MV_02_video.mp4")

    # Reading the first frame of the video
    ret, first_frame = vid.read()
    if ret:
        cv2.imshow("1st frame", first_frame)
        # cv2.waitKey()

        # Converting color to gray image
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Extracting interest points in the first frame of the video
        features = cv2.goodFeaturesToTrack(gray, 200, 0.3, 7)
        print("\n", len(features))
        print("\n", features[0:5, :, :])

        # Extracting interest points with subpixel accuracy in the first frame
        subpixel_features = cv2.cornerSubPix(gray, features, (11, 11), (-1, -1), criteria=criteria)

        print("\n", len(subpixel_features))
        print("\n", subpixel_features[0:5, :, :])

        index = np.arange(len(subpixel_features))
        tracks_to_visualize = {}

        for i in range(len(subpixel_features)):
            tracks_to_visualize[index[i]] = {0: subpixel_features[i]}

        print("\ntracks: \n", tracks_to_visualize)

        frame = 0
        while ret:
            ret, img = vid.read()

            if not ret:
                break

            frame += 1
            old_img = gray
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if len(subpixel_features) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, gray, subpixel_features, None)

                # visualise points
                for i in range(len(st)):
                    if st[i]:
                        cv2.circle(img, (p1[i, 0, 0], p1[i, 0, 1]), 2, (0, 0, 255), 2)
                        cv2.line(img, (subpixel_features[i, 0, 0], subpixel_features[i, 0, 1]), (int(subpixel_features[i, 0, 0] + (p1[i][0, 0] - subpixel_features[i, 0, 0]) * 5), int(subpixel_features[i, 0, 1] + (p1[i][0, 1] - subpixel_features[i, 0, 1]) * 5)), (0, 0, 255), 2)

                subpixel_features = p1[st == 1].reshape(-1, 1, 2)
                index = index[st.flatten() == 1]

            # refresh features, if too many lost
            if len(subpixel_features) < 100:
                features = cv2.goodFeaturesToTrack(gray, 200 - len(subpixel_features), 0.3, 7)
                new_subpixel_features = cv2.cornerSubPix(gray, features, (11, 11), (-1, -1), criteria=criteria)
                for i in range(len(new_subpixel_features)):
                    if np.min(np.linalg.norm((subpixel_features - new_subpixel_features[i]).reshape(len(subpixel_features), 2), axis=1)) > 10:
                        subpixel_features = np.append(subpixel_features, new_subpixel_features[i].reshape(-1, 1, 2), axis=0)
                        index = np.append(index, np.max(index) + 1)

            # update tracks
            for i in range(len(subpixel_features)):
                if index[i] in tracks_to_visualize:
                    tracks_to_visualize[index[i]][frame] = subpixel_features[i]
                else:
                    tracks_to_visualize[index[i]] = {frame: subpixel_features[i]}

            # visualise last frames of active tracks
            for i in range(len(index)):
                for f in range(frame - 20, frame):
                    if (f in tracks_to_visualize[index[i]]) and (f + 1 in tracks_to_visualize[index[i]]):
                        cv2.line(img,
                                 (tracks_to_visualize[index[i]][f][0, 0], tracks_to_visualize[index[i]][f][0, 1]),
                                 (tracks_to_visualize[index[i]][f + 1][0, 0], tracks_to_visualize[index[i]][f + 1][0, 1]),
                                 (0, 255, 0), 1)

            k = cv2.waitKey(10)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break

            cv2.imshow("feature track", img)
            # cv2.waitKey()

        cv2.destroyAllWindows()
        print("\nlen of tracks: ", len(tracks_to_visualize))
        print("tot frames: ", frame)
        return tracks_to_visualize, mtx, frame, image_points, object_points


def extract_frames(filename, frames):
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame = 0
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            break
        if frame in frames:
            result[frame] = img
        frame += 1
        if frame > last_frame:
            break

    return result


# Calculate the epipole points using Fundamental matrix
def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)
    e1 = V[2, :]

    U,S,V = np.linalg.svd(F.T)
    e2 = V[2, :]

    return e1, e2


# Function to visualize the tracks in both first frame and last frame
def visualize_tracks_in_both_frames(image1, image2, tracks_to_visualize):
    print("in visualization")
    index = np.arange(len(tracks_to_visualize))

    print("\n", tracks_to_visualize[0][0][0])

    for i in range(len(index)):
        cv2.line(image1,
                 (int(tracks_to_visualize[i][0][0]), int(tracks_to_visualize[i][0][1])),
                 (int(tracks_to_visualize[i][1][0]), int(tracks_to_visualize[i][1][1])),
                 (0, 255, 0), 2)

        cv2.line(image2,
                 (int(tracks_to_visualize[i][0][0]), int(tracks_to_visualize[i][0][1])),
                 (int(tracks_to_visualize[i][1][0]), int(tracks_to_visualize[i][1][1])),
                 (0, 255, 0), 2)

    cv2.imshow("frame 0: ", image1)
    cv2.imshow("frame 30: ", image2)
    cv2.waitKey()


def calculate_epipolar_line(F, x, width, height):
    l = np.matmul(F, x)
    l1 = np.cross([0,0,1],[width-1,0,1])
    l2 = np.cross([0,0,1],[0,height-1,1])
    l3 = np.cross([width-1,0,1],[width-1,height-1,1])
    l4 = np.cross([0,height-1,1],[width-1,height-1,1])
    x1 = np.cross(l,l1)
    x2 = np.cross(l,l2)
    x3 = np.cross(l,l3)
    x4 = np.cross(l,l4)
    x1 /= x1[2]
    x2 /= x2[2]
    x3 /= x3[2]
    x4 /= x4[2]
    result = []
    if (x1[0]>=0) and (x1[0]<=width):
        result.append(x1)
    if (x2[1]>=0) and (x2[1]<=height):
        result.append(x2)
    if (x3[1]>=0) and (x3[1]<=height):
        result.append(x3)
    if (x4[0]>=0) and (x4[0]<=width):
        result.append(x4)
    return result[0], result[1]


# Fundamental matrix
def task2(alltracks, no_of_frames):
    print("task2")

    first_frame = 0
    last_frame = no_of_frames

    # A) Stores the tracks visible in both first and last frame (common tracks)
    correspondences = []

    # Filter the tracks visible in both first and last frames
    for track in alltracks:
        if first_frame in alltracks[track] and last_frame in alltracks[track]:
            x1 = [alltracks[track][first_frame][0, 1], alltracks[track][first_frame][0, 0], 1]
            x2 = [alltracks[track][last_frame][0, 1], alltracks[track][last_frame][0, 0], 1]
            correspondences.append([np.array(x1), np.array(x2)])

    images = extract_frames("../../../Assignment_MV_02_video.mp4", [0, last_frame])

    # visualize_tracks_in_both_frames(images[0], images[30], correspondences)

    # B) Calculate mean of the feature coordinates
    mu = np.mean(np.array(correspondences)[:, 0, :2], axis=0)
    muprime = np.mean(np.array(correspondences)[:, 1, :2], axis=0)

    # Calculate standard deviations of the feature coordinates
    sig = np.std(np.array(correspondences)[:, 0, :2], axis=0)
    sigprime = np.std(np.array(correspondences)[:, 1, :2], axis=0)

    print("\nmu: ", mu)
    print("\nmuPrime: ", muprime)

    print("\nstd: ", sig)
    print("\nstdPrime: ", sigprime)

    T = np.array([[1/sig[1], 0, -mu[1]/sig[1]],
                  [0, 1/sig[0], -mu[0]/sig[0]],
                  [0, 0, 1]
                  ]
                 )

    Tprime = np.array([[1 / sigprime[1], 0, -muprime[1] / sigprime[1]],
                       [0, 1 / sigprime[0], -muprime[0] / sigprime[0]],
                       [0, 0, 1]
                       ]
                      )
    # Normalizing the features
    yi = (T@np.array(correspondences)[:, 0, :].T).T
    yiprime = (Tprime@np.array(correspondences)[:, 1, :].T).T

    indices = [i for i in range(len(correspondences))]

    # Cxx = np.identity(3)
    Cxx = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

    best_inliner_test_statistic = np.inf
    best_outlier_count = len(correspondences)
    best_fundamental_matrix = None
    best_inliers = None

    # Iterating for 10000 times to find the best fundamental matrix
    for i in range(10000):
        outliers = list()
        inliners = list()
        inliner_test_statistics = 0

        # Selecting 8 random correspondences
        random_8 = random.sample(indices, 8)
        remaining_indices = list(set(indices) - set(random_8))

        A = np.zeros((0, 9))
        for y, yprime in zip(yi[random_8, :], yiprime[random_8, :]):
            ai = np.kron(y.T, yprime.T)
            A = np.append(A, [ai], axis=0)

        u, s, v = np.linalg.svd(A)
        Fhat = v[8, :].reshape(3, 3).T

        U, S, V = np.linalg.svd(Fhat)
        Fhat = np.matmul(U, np.matmul(np.diag([S[0], S[1], 0]), V))

        F = np.matmul(Tprime.T, np.matmul(Fhat, T))
        best_fundamental_matrix = F

        j = 0
        for xi, xiprime in np.array(correspondences)[remaining_indices, :, :]:
            gi = np.matmul(np.matmul(xiprime.reshape(3, 1).T, F), xi.reshape(3, 1))
            variance = np.matmul(np.matmul(np.matmul(np.matmul(xiprime.reshape(3, 1).T, F), Cxx), F.T), xiprime.reshape(3, 1)) + np.matmul(np.matmul(np.matmul(np.matmul(xi.reshape(3, 1).T, F.T), Cxx), F), xi.reshape(3, 1))

            Ti = gi**2 / variance

            if Ti > 6.635:
                outliers.append(j)
            else:
                inliners.append([xi, xiprime])
                # Sum up all test statistics over all inliners
                inliner_test_statistics += Ti

            j += 1

        if len(outliers) > 0:
            print("{} outliers found in iteration {}".format(len(outliers), i+1))

        if len(outliers) == best_outlier_count:
            if inliner_test_statistics < best_inliner_test_statistic:
                best_inliner_test_statistic = inliner_test_statistics
                print("Discarding previous fundamental matrix as the test statics sum is lesser than the previous one")
                best_fundamental_matrix = F
                best_inliers = inliners

        elif len(outliers) < best_outlier_count:
            best_outlier_count = len(outliers)
            best_fundamental_matrix = F
            best_inliers = inliners

    print("Best outlier counts after 10000 iterations : ", best_outlier_count)
    print("Best Fundamental matrix : \n", best_fundamental_matrix)
    print("Total inliers after 10000 iterations: ", len(best_inliers))

    # Epipoles
    e1, e2 = calculate_epipoles(best_fundamental_matrix)

    print("epipoles: \n")
    print(e1)
    print("\n")
    print(e2)
    print("\n")
    print(e1 / e1[2])
    print(e2 / e2[2])

    # print("images keys: ", images.keys())

    width = images[first_frame].shape[1]
    height = images[last_frame].shape[0]

    x = np.array([0.5 * width,
                  0.5 * height,
                  1])

    cv2.circle(images[0], (int(e1[0] / e1[2]), int(e1[1] / e1[2])), 3, (0, 0, 255), 2)
    cv2.circle(images[30], (int(e2[0] / e2[2]), int(e2[1] / e2[2])), 3, (0, 0, 255), 2)

    x1, x2 = calculate_epipolar_line(best_fundamental_matrix, x, width, height)

    cv2.circle(images[first_frame], (int(x[0] / x[2]), int(x[1] / x[2])), 3, (0, 255, 0), 2)
    cv2.line(images[last_frame], (int(x1[0] / x1[2]), int(x1[1] / x1[2])), (int(x2[0] / x2[2]), int(x2[1] / x2[2])),
             (0, 255, 0), 2)

    cv2.imshow("img1", images[0])
    cv2.imshow("img2", images[30])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return best_fundamental_matrix, best_inliers


# Essential matrix
def task3(inliers, K, F, no_of_frames):
    number_of_frames = no_of_frames
    # Checking if determinant of Fundamental matrix is zero
    print("Determinant of Fundamental matrix: ", np.linalg.det(F))

    Ehat = np.matmul(np.matmul(K.T, F), K)
    print("\n Essential matrix estimation: \n", Ehat)

    U, S, V = np.linalg.svd(Ehat)
    # E = np.matmul(U, np.matmul(np.diag([S[0], S[1], 0]), V))

    print("\nU: \n", U)

    # Making sure the determinants of rotation matrices are positive
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    if np.linalg.det(V) < 0:
        V[2, :] *= -1

    print("Determinant of U: ", np.linalg.det(U))

    print("\nV: \n", V)

    print("\nDeterminant of V: ", np.linalg.det(V))

    # Singular values of E.
    print("\nS: \n", S)

    # non zero singular values must be Identical, averaging out the values
    s = (S[0] + S[1]) / 2

    E = np.matmul(U, np.matmul(np.diag([s, s, 0]), V.T))
    print("\nEssential matrix: \n", E)

    # Checking if the determinant is equal to 0
    print("Determinant of E: ", np.linalg.det(E))

    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    # Convert speed 50kmph into meters per second
    speed = (50*1000)/3600
    beta = (speed * no_of_frames)/30

    UZUT = (np.matmul(np.matmul(U, Z), U.T))

    SRTt1 = beta*UZUT
    SRTt2 = -beta * UZUT  # or just -SRTt1 (both are same)

    RTt1 = np.array([SRTt1[2, 1], SRTt1[0, 2], SRTt1[1, 0]])
    RTt2 = np.array([SRTt2[2, 1], SRTt2[0, 2], SRTt2[1, 0]])

    print("\nSRTt1: \n", SRTt1)
    print("\nSRTt2: \n", SRTt2)

    # Rotation matrices
    RT1 = np.matmul(np.matmul(U, W), V.T)
    RT2 = np.matmul(np.matmul(U, W.T), V.T)
    rotation_matrices = [RT1, RT2]

    print("\nRotation matrix 1: \n", RT1)
    print("\nRotation matrix 2: \n", RT2)

    # Checking the determinants of rotation matrix, must be positive
    print("\nDeterminant of Rotation Matrix1: ", np.linalg.det(RT1))
    print("\nDeterminant of Rotation Matrix2: ", np.linalg.det(RT2))

    # Translation matrices
    t1 = np.matmul(np.linalg.inv(RT1), RTt1)
    t2 = np.matmul(np.linalg.inv(RT2), RTt2)
    translation_matrices = [t1, t2]

    print("\nTranslation vector 1: ", t1)
    print("\nTranslation vector 2: ", t2)

    ''' Calculate for each inlier feature correspondence determined in task 2 and each potential solution calculated 
    in the previous subtask the directions 𝒎 and 𝒎′ of the 3d lines originating from the centre of projection towards the 3d points'''

    '''calculate the unknown distances 𝜆 and 𝜇 by solving the linear equation system
    (𝒎𝑻𝒎    −𝒎𝑻𝑹𝒎′  (𝜆) =  (𝒕𝑻𝒎)
    𝒎𝑻𝑹𝒎′   −𝒎′𝑻𝒎′) (𝜇) = (𝒕𝑻𝑹𝒎′)'''
    all_3d_coordinates = {}     # dictionary where key is a tuple (rotation matrix, translation matrix)

    '''dictionary where key is a tuple (rotation matrix, translation matrix) 
        which helps to determine which rotational and translational matrix to use and also
        discard the points which are behind the frames'''

    all_counts = {}

    for i, rotation_matrix in enumerate(rotation_matrices):
        for j, translation_matrix in enumerate(translation_matrices):
            best_points_count = 0
            coordinates_3d = list()
            for x1, x2 in inliers:
                # compute the directions
                m = np.matmul(np.linalg.inv(K), x1)
                mprime = np.matmul(np.linalg.inv(K), x2)

                # Components of linear equations to be solved
                mTm = np.matmul(m.T, m)
                mTRmprime = np.matmul(m.T, np.matmul(rotation_matrix, mprime))
                mprimeTmprime = np.matmul(mprime.T, mprime)
                tTm = np.matmul(translation_matrix.T, m)
                tTRmprime = np.matmul(translation_matrix.T, np.matmul(rotation_matrix, mprime))

                lambda_mu = np.linalg.solve([[mTm, -mTRmprime], [mTRmprime, -mprimeTmprime]], [tTm, tTRmprime])
                if lambda_mu[0] > 0 and lambda_mu[1] > 0:
                    best_points_count += 1
                    xlambda = lambda_mu[0] * m
                    xmu = translation_matrix + np.multiply(lambda_mu[1], np.matmul(rotation_matrix, mprime))
                    coordinates_3d.append([xlambda, xmu])

            all_3d_coordinates[(i, j)] = coordinates_3d
            all_counts[(i, j)] = best_points_count

    print("\nall_3d_coordinates: \n", all_3d_coordinates)
    print("\nall counts: \n", all_counts)

    best_coordinates_count = max(list(all_counts.values()))
    print("best_coordinates_count: ", best_coordinates_count)
    best_coordinates_count_index = np.argmax(list(all_counts.values()))
    print("best_coordinates_count_index: ", best_coordinates_count_index)
    best_coordinates_count_key = list(all_counts.keys())[best_coordinates_count_index]
    print("best count key: ", best_coordinates_count_key)
    points_infront_both_frames = all_3d_coordinates[best_coordinates_count_key]
    print("\nTo plot: \n", points_infront_both_frames)
    print("\nBest rotational matrix: \n", rotation_matrices[best_coordinates_count_key[0]])
    print("\nBest translational matrix: \n", translation_matrices[best_coordinates_count_key[1]])

    print(np.array(points_infront_both_frames).shape)

    # Display all 3d points
    ax = Axes3D(plt.figure())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x, y, z = np.array(points_infront_both_frames)[:, 0, 0], np.array(points_infront_both_frames)[:, 0, 1], np.array(points_infront_both_frames)[:, 0, 2]
    ax.scatter3D(x, y, z, marker='o', c='red')

    ax.plot([0.], [0.], [0.], marker='X', c='blue')

    ax.plot(translation_matrices[best_coordinates_count_key[1]][0], translation_matrices[best_coordinates_count_key[1]][1], translation_matrices[best_coordinates_count_key[1]][2], marker='o', c='green')

    plt.show()


if __name__ == "__main__":
    tracks, K, no_of_frames, image_points, object_points = task1()
    print("image points: ", image_points)
    print("\n object points: ", object_points)
    # F, inliers = task2(tracks, no_of_frames)
    # print("total inliers: ", len(inliers))
    # print("\nBest inliers: \n", inliers)
    # task3(inliers, K, F, no_of_frames)
