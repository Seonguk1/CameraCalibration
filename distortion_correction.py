import cv2 as cv
import numpy as np

video_file = 'data/chessboard.mp4'

K = np.array([[947.94697884, 0., 440.16542092],
              [0., 993.15217813, 866.77735144],
              [0., 0., 1.]])
dist_coeff = np.array([ 0.01121081,  0.16799448, -0.02977654,  0.03465663,  0.00860905])

video = cv.VideoCapture(video_file)
assert video.isOpened(), f'Cannot open the video: {video_file}'

show_rectify = True
map1, map2 = None, None
saved_original = False
saved_rectified = False

while True:
    valid, img = video.read()
    if not valid:
        break

    img_show = img.copy()

    if not saved_original:
        cv.imwrite('original_sample.jpg', img)
        saved_original = True

    info = 'Original'
    if show_rectify:
        if map1 is None or map2 is None:
            map1, map2 = cv.initUndistortRectifyMap(
                K, dist_coeff, None, None,
                (img.shape[1], img.shape[0]),
                cv.CV_32FC1
            )
        img_show = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        info = 'Rectified'

        if not saved_rectified:
            cv.imwrite('rectified_sample.jpg', img_show)
            saved_rectified = True

    cv.putText(img_show, info, (10, 25),
               cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv.imshow('Distortion correction', img_show)

    key = cv.waitKey(10)
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        show_rectify = not show_rectify

video.release()
cv.destroyAllWindows()