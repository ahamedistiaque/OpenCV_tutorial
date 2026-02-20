#probles ase

import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture("traffic2.mp4")
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (720, 500))
    cv2.imshow("Select ROI - Press ENTER or SPACE after selection", frame)

    # Let user select ROI manually (on resized frame)
    track_window = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    if track_window == (0,0,0,0):
        print("No ROI selected, exiting.")
        return

    x, y, w, h = map(int, track_window)

    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Mask: keep only reasonable saturation and brightness (tune if needed)
    mask = cv2.inRange(hsv_roi,
                       np.array((0., 40., 40.)),
                       np.array((180., 255., 255.)))

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        frame = cv2.resize(frame, (720, 500))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        # MeanShift tracking
        ret_ms, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = map(int, track_window)
        mean_frame = frame.copy()
        cv2.rectangle(mean_frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(mean_frame, "MeanShift", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # CamShift tracking
        ret_cs, track_window_cs = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret_cs)
        pts = np.intp(pts)
        cam_frame = frame.copy()
        cv2.polylines(cam_frame, [pts], True, (255,0,0), 2)
        cv2.putText(cam_frame, "CamShift", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("MeanShift Tracking", mean_frame)
        cv2.imshow("CamShift Tracking", cam_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break

        # Optional: check if windows are closed manually
        if cv2.getWindowProperty("MeanShift Tracking", cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty("CamShift Tracking", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed manually. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
