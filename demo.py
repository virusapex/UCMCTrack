import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper


# Define a Detection class containing id, bounding box coordinates, confidence, and class
class Detection:
    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def __str__(self):
        return (
            f'd{self.id}, bb_box:[{self.bb_left},{self.bb_top},{self.bb_width},{self.bb_height}], '
            f'conf={self.conf:.2f}, class{self.det_class}, uv:[{self.bb_left + self.bb_width / 2:.0f},'
            f'{self.bb_top + self.bb_height:.0f}], mapped to:[{self.y[0, 0]:.1f},{self.y[1, 0]:.1f}]'
        )

    def __repr__(self):
        return self.__str__()


# Detector class used to get detection results from YOLO detector
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.model = YOLO('pretrained/yolo11x.pt')

    def get_dets(self, img, conf_thresh=0):
        dets = []

        # Use YOLO to get detections directly (no need for BGR to RGB conversion)
        results = self.model(img, imgsz=1280)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if (w <= 10 and h <= 10) or conf <= conf_thresh:
                continue

            # Create a new Detection object
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1

            dets.append(det)

        return dets


def main(args):
    cap = cv2.VideoCapture(args.video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = cv2.VideoWriter(
        'output/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
    )

    # Open a cv window with specified height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    tracker = UCMCTrack(
        args.a,
        args.a,
        args.wx,
        args.wy,
        args.vmax,
        args.cdt,
        fps,
        "MOT",
        args.high_score,
        False,
        None,
    )

    # Loop through video frames
    frame_id = 1
    while True:
        frame_start_time = time.perf_counter()

        ret, frame_img = cap.read()
        if not ret:
            break
        read_time = time.perf_counter() - frame_start_time  # Time to read frame

        det_start_time = time.perf_counter()
        dets = detector.get_dets(frame_img, args.conf_thresh)
        det_time = time.perf_counter() - det_start_time  # Time for detection

        track_start_time = time.perf_counter()
        tracker.update(dets, frame_id)
        track_time = time.perf_counter() - track_start_time  # Time for tracking

        vis_start_time = time.perf_counter()
        for det in dets:
            # Draw detection boxes
            if det.track_id > 0:
                cv2.rectangle(
                    frame_img,
                    (int(det.bb_left), int(det.bb_top)),
                    (
                        int(det.bb_left + det.bb_width),
                        int(det.bb_top + det.bb_height),
                    ),
                    (0, 255, 0),
                    2,
                )
                # Draw detection ID
                cv2.putText(
                    frame_img,
                    str(det.track_id),
                    (int(det.bb_left), int(det.bb_top)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        vis_time = time.perf_counter() - vis_start_time  # Time for visualization

        total_time = time.perf_counter() - frame_start_time  # Total time per frame
        current_fps = 1 / total_time if total_time > 0 else 0

        # Display timing information on the frame
        cv2.putText(
            frame_img,
            f"Frame ID: {frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame_img,
            f"Read Time: {read_time*1000:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame_img,
            f"Detection Time: {det_time*1000:.1f} ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame_img,
            f"Tracking Time: {track_time*1000:.1f} ms",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame_img,
            f"Visualization Time: {vis_time*1000:.1f} ms",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame_img,
            f"Total Time: {total_time*1000:.1f} ms",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame_img,
            f"FPS: {current_fps:.1f}",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        frame_id += 1

        # Display the current frame
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)

        video_out.write(frame_img)

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default="demo/demo.mp4", help='video file name')
parser.add_argument(
    '--cam_para', type=str, default="demo/cam_para.txt", help='camera parameter file name'
)
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument(
    '--conf_thresh', type=float, default=0.01, help='detection confidence threshold'
)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
