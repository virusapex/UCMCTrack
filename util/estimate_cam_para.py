import argparse
import cv2
import numpy as np


class CameraParameterAdjuster:
    def __init__(self, img, output_file):
        self.output_file = output_file
        self.original_img = img
        self.height, self.width = img.shape[:2]

        # Initialize intrinsic matrix Ki
        self.Ki = np.array([
            [1000,    0, self.width / 2],
            [   0, 1000, self.height / 2],
            [   0,    0,             1]
        ])

        # Add an extra column of zeros to Ki for compatibility
        self.Ki = np.hstack((self.Ki, np.zeros((3, 1))))

        # Initialize extrinsic matrix Ko as identity
        self.Ko = np.eye(4)

        # Parameters initialized to default values
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
        self.focal = 0
        self.tz = 0

        self.setup_ui()

    def setup_ui(self):
        cv2.namedWindow('Values')
        cv2.namedWindow('CamParaSettings')

        # Create trackbars
        cv2.createTrackbar('theta_x', 'CamParaSettings', 250, 500, self.update_theta_x)
        cv2.createTrackbar('theta_y', 'CamParaSettings', 250, 500, self.update_theta_y)
        cv2.createTrackbar('theta_z', 'CamParaSettings', 500, 1000, self.update_theta_z)
        cv2.createTrackbar('focal', 'CamParaSettings', 100, 500, self.update_focal)
        cv2.createTrackbar('Tz', 'CamParaSettings', 30, 500, self.update_tz)

    def update_theta_x(self, val):
        self.theta_x = (val - 250) / 10.0
        self.display_values()

    def update_theta_y(self, val):
        self.theta_y = (val - 250) / 10.0
        self.display_values()

    def update_theta_z(self, val):
        self.theta_z = (val - 500) / 5.0
        self.display_values()

    def update_focal(self, val):
        self.focal = (val - 100) * 5
        self.display_values()

    def update_tz(self, val):
        self.tz = (val - 30) * 0.04
        self.display_values()

    def display_values(self):
        display_img = np.zeros((400, 300), dtype=np.uint8)
        params = [
            f"theta_x: {self.theta_x:.2f}",
            f"theta_y: {self.theta_y:.2f}",
            f"theta_z: {self.theta_z:.2f}",
            f"focal: {self.focal}",
            f"Tz: {self.tz:.2f}"
        ]
        for idx, text in enumerate(params):
            cv2.putText(display_img, text, (10, 60 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow('Values', display_img)

    def compute_uv(self, x, y, Ki, Ko):
        uvw = Ki @ Ko @ np.array([x, y, 0, 1])
        uv = uvw[:2] / uvw[2]
        return tuple(map(int, uv))

    def run(self):
        while True:
            Ki = self.Ki.copy()
            Ko = self.Ko.copy()

            # Update rotation matrices
            Rx = cv2.Rodrigues(np.array([np.radians(self.theta_x), 0, 0]))[0]
            Ry = cv2.Rodrigues(np.array([0, np.radians(self.theta_y), 0]))[0]
            Rz = cv2.Rodrigues(np.array([0, 0, np.radians(self.theta_z)]))[0]
            R = Ko[:3, :3] @ Rx @ Ry @ Rz
            Ko[:3, :3] = R

            # Update translation and focal length
            Ko[2, 3] += self.tz
            Ki[0, 0] += self.focal
            Ki[1, 1] += self.focal

            # Draw points on the image
            img_copy = self.original_img.copy()
            for x in np.arange(0, 10, 0.5):
                for y in np.arange(-5, 5, 0.5):
                    try:
                        u, v = self.compute_uv(x, y, Ki, Ko)
                        cv2.circle(img_copy, (u, v), 3, (0, 255, 0), -1)
                    except Exception:
                        continue  # Skip points that result in invalid pixel coordinates

            # Display the image
            resized_img = cv2.resize(img_copy, (self.width // 2, self.height // 2))
            cv2.imshow('img', resized_img)
            if cv2.waitKey(50) == ord('q'):
                self.save_parameters(Ki, Ko)
                break

    def save_parameters(self, Ki, Ko):
        with open(self.output_file, 'w') as f:
            f.write("RotationMatrices\n")
            for row in Ko[:3, :3]:
                f.write(' '.join(f"{val:.6f}" for val in row) + "\n")
            f.write("\nTranslationVectors\n")
            f.write(' '.join(f"{int(val * 1000)}" for val in Ko[:3, 3]) + "\n")
            f.write("\nIntrinsicMatrix\n")
            for row in Ki[:3, :3]:
                f.write(' '.join(f"{int(val)}" for val in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Estimate camera parameters from a video frame.')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--cam_para', type=str, required=True, help='Path to save the camera parameters file')
    args = parser.parse_args()

    # Capture the 100th frame from the video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.video}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 99)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Cannot read frame 100 from {args.video}")
        return

    # Initialize and run the adjuster
    adjuster = CameraParameterAdjuster(frame, args.cam_para)
    adjuster.run()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
