import cv2 as cv
import numpy as np

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    # Open a video
    video = cv.VideoCapture(video_file)
    
    # Select images 
    img_select = []
    print("스페이스바를 눌러 체스보드 이미지를 캡처하세요. 충분히 캡처했다면 ESC를 누르세요.")
    
    while True:
        valid, img = video.read()
        if not valid:
            break
            
        display = img.copy()
        cv.putText(display, f"Captured: {len(img_select)}", (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        cv.imshow('Camera Calibration', display)
        
        key = cv.waitKey(wait_msec)
        if key == 27: # ESC
            break
        elif key == ord(' ') or select_all: # 스페이스바를 누르거나 select_all이 True일 때
            img_select.append(img)
            print(f"이미지 캡처됨: 총 {len(img_select)}장")
            
    video.release()
    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images [cite: 871]
    img_points = [] 
    for img in images: 
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern) 
        if complete:
            img_points.append(pts) 
            
    assert len(img_points) > 0, 'There is no set of complete chessboard points!' 

    # Prepare 3D points of the chess board [cite: 880]
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32` [cite: 882]

    # Calibrate the camera [cite: 883]
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)


if __name__ == '__main__':
    # 촬영한 체스보드 동영상 경로
    video_file = 'data/chessboard.mp4' # 본인의 파일명으로 변경하세요
    board_pattern = (6, 8)   # 체스보드의 내부 코너 개수 (가로, 세로) [cite: 825]
    board_cellsize = 0.025   # 실제 체스보드 한 칸의 크기 (미터 단위, 예: 25mm = 0.025m)
    
    images = select_img_from_video(video_file, board_pattern)
    
    if len(images) > 0:
        rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(images, board_pattern, board_cellsize)
        
        # README.md 작성을 위한 결과 출력 [cite: 1133]
        print("\n## Camera Calibration Results") 
        print(f"* RMS error = {rms}") 
        print(f"* Camera matrix (K) = \n{K}") 
        print(f"* Distortion coefficient (k1, k2, p1, p2, k3, ...) = \n{dist_coeff.flatten()}")