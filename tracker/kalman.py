import numpy as np
from numba import njit
from enum import Enum

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted = 2

class NumbaKalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F        # State transition matrix
        self.H = H        # Measurement matrix
        self.Q = Q        # Process noise covariance
        self.R = R        # Measurement noise covariance
        self.P = P        # Estimate error covariance
        self.x = x        # State estimate

    @staticmethod
    # @njit
    def predict(F, x, P, Q):
        x = F @ x
        P = F @ P @ F.T + Q
        return x, P

    @staticmethod
    # @njit
    def update(x, P, z, R, H):
        y = z - H @ x  # Measurement residual
        S = H @ P @ H.T + R  # Residual covariance

        # Add small epsilon to diagonal for numerical stability
        epsilon = 1e-6
        S[0, 0] += epsilon
        S[1, 1] += epsilon

        # Compute Kalman Gain
        K = P @ H.T @ np.linalg.inv(S)

        # Update state estimate and covariance
        x = x + K @ y
        P = P - K @ H @ P

        return x, P

class KalmanTracker:
    """Kalman Tracker class with Numba-optimized Kalman filter."""

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w, h, class_label, dt=1 / 30):
        # State transition matrix
        F = np.array(
            [
                [1.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64
        )

        # Measurement matrix
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64
        )

        # Process noise covariance
        G = np.array(
            [
                [0.5 * dt**2, 0.0],
                [dt, 0.0],
                [0.0, 0.5 * dt**2],
                [0.0, dt],
            ],
            dtype=np.float64
        )
        Q0 = np.diag([wx, wy]).astype(np.float64)
        Q = G @ Q0 @ G.T

        # Initial estimate error covariance
        P = np.diag([1.0, (vmax**2) / 3.0, 1.0, (vmax**2) / 3.0]).astype(np.float64)

        # Initial state estimate
        x = np.zeros((4,), dtype=np.float64)
        x[0] = y[0, 0]
        x[2] = y[1, 0]

        # Create NumbaKalmanFilter instance
        self.kf = NumbaKalmanFilter(F, H, Q, R.astype(np.float64), P, x)

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h

        self.class_label = class_label
        self.status = TrackStatus.Tentative

    def update(self, y, R):
        """Update the state with observed data."""
        y = y.flatten().astype(np.float64)
        R = R.astype(np.float64)
        self.kf.x, self.kf.P = self.kf.update(self.kf.x, self.kf.P, y, R, self.kf.H)

    def predict(self):
        """Predict the next state."""
        self.kf.x, self.kf.P = self.kf.predict(self.kf.F, self.kf.x, self.kf.P, self.kf.Q)
        self.age += 1
        return self.kf.H @ self.kf.x

    def get_state(self):
        """Get the current state estimate."""
        return self.kf.x
