import cv2 as cv
import numpy as np


def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), f'Cannot open the video: {video_file}'

    # Select images
    img_select = []

    while True:
        valid, img = video.read()
        if not valid:
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)

        img_display = img.copy()
        if complete:
            cv.drawChessboardCorners(img_display, board_pattern, pts, complete)
            cv.putText(img_display, 'Chessboard found', (10, 25),
                       cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

            if select_all:
                img_select.append(img.copy())
        else:
            cv.putText(img_display, 'Chessboard not found', (10, 25),
                       cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))

        cv.imshow('Select calibration images', img_display)

        if select_all:
            key = cv.waitKey(wait_msec)
            if key == 27:  # ESC
                break
        else:
            key = cv.waitKey(0)
            if key == 27:  # ESC
                break
            elif key == 32 and complete:  # Space
                img_select.append(img.copy())

    video.release()
    cv.destroyAllWindows()
    return img_select


def calib_camera_from_chessboard(images, board_pattern, board_cellsize,
                                 K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    gray = None

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)

    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)  # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)


if __name__ == '__main__':
    video_file = 'data/chessboard.mp4'
    board_pattern = (7, 9)
    board_cellsize = 1.0
    select_all = True

    # Select calibration images from the video
    images = select_img_from_video(video_file, board_pattern, select_all=select_all, wait_msec=30)
    print(f'# Selected images = {len(images)}')

    # Calibrate the camera
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(
        images, board_pattern, board_cellsize
    )

    # Print calibration results
    print('\n## Camera Calibration Results')
    print(f'* The number of applied images = {len(images)}')
    print(f'* RMS error = {rms}')
    print('* Camera matrix (K) =')
    print(K)
    print('* Distortion coefficient (k1, k2, p1, p2, k3, ...) =')
    print(dist_coeff.flatten())

    print(f'* fx = {K[0, 0]}')
    print(f'* fy = {K[1, 1]}')
    print(f'* cx = {K[0, 2]}')
    print(f'* cy = {K[1, 2]}')

    # Save calibration results
    np.savez('calibration_result.npz', K=K, dist_coeff=dist_coeff, rms=rms)