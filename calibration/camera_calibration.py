"""
calibration/camera_calibration.py

OpenCV checkerboard camera calibration script.

Run this once before your first drive to get accurate camera intrinsics.
These intrinsics are used by the depth pipeline for metric scale recovery.

Usage:
    python calibration/camera_calibration.py

Instructions:
    1. Print or display the checkerboard pattern (calibration/checkerboard_9x6.png)
    2. Hold the pattern at various angles and distances in front of the camera
    3. Press SPACE to capture a frame, Q to quit and compute calibration
    4. Aim for 15–25 frames from diverse angles (tilt, rotate, distance)
    5. Results saved to calibration/camera_params.json
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Checkerboard inner corners (pattern must match what you print)
BOARD_ROWS = 6  # Inner corner rows (not squares — one less than square count)
BOARD_COLS = 9  # Inner corner columns
SQUARE_SIZE_MM = 25.0  # Physical size of each square in mm


def calibrate_camera():
    """Run interactive checkerboard calibration from webcam."""
    # 3D object points in the checkerboard plane (Z=0)
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE_MM

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera. Check camera connection.")
        sys.exit(1)

    logger.info("Camera calibration started.")
    logger.info(f"Board: {BOARD_COLS}x{BOARD_ROWS} inner corners, {SQUARE_SIZE_MM}mm squares.")
    logger.info("SPACE = capture frame | Q = quit & compute calibration")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), None)

        display = frame.copy()
        status_color = (0, 180, 0) if found else (0, 0, 200)

        if found:
            cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners, found)
            cv2.putText(display, f"Board FOUND | Captures: {frame_count} | SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            cv2.putText(display, f"Board NOT found | Captures: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        cv2.imshow("PULSE — Camera Calibration (SPACE=capture, Q=quit)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        if key == ord(" ") and found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            frame_count += 1
            logger.info(f"Captured frame {frame_count}")

            if frame_count >= 25:
                logger.info("25 frames captured — auto-proceeding to calibration.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if frame_count < 5:
        logger.error(f"Only {frame_count} frames captured — need at least 5. Aborting.")
        sys.exit(1)

    logger.info(f"Computing calibration from {frame_count} frames...")

    h, w = gray.shape
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    if not ret:
        logger.error("Calibration failed — unexpected error from OpenCV.")
        sys.exit(1)

    # Reprojection error
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                           camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    logger.info(f"Calibration complete. Reprojection error: {mean_error:.4f} px")
    if mean_error > 1.0:
        logger.warning("Reprojection error > 1px — consider recalibrating with more diverse angles.")

    params = {
        "image_width":  w,
        "image_height": h,
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "k1": float(dist_coeffs[0, 0]),
        "k2": float(dist_coeffs[0, 1]),
        "p1": float(dist_coeffs[0, 2]),
        "p2": float(dist_coeffs[0, 3]),
        "k3": float(dist_coeffs[0, 4]) if dist_coeffs.shape[1] > 4 else 0.0,
        "reprojection_error_px": round(mean_error, 4),
        "frames_used": frame_count,
        "board_rows":  BOARD_ROWS,
        "board_cols":  BOARD_COLS,
        "square_size_mm": SQUARE_SIZE_MM,
    }

    output_path = Path(__file__).parent / "camera_params.json"
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"Calibration saved to {output_path}")
    logger.info(f"  fx={params['fx']:.1f}, fy={params['fy']:.1f}")
    logger.info(f"  cx={params['cx']:.1f}, cy={params['cy']:.1f}")
    logger.info("Set CALIBRATION_FILE=calibration/camera_params.json in backend/.env")


if __name__ == "__main__":
    calibrate_camera()
