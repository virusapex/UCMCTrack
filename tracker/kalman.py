from filterpy.kalman import KalmanFilter
import numpy as np
from enum import Enum


class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted = 2


class KalmanTracker:
    """Kalman Tracker class with class label integration."""

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w, h, class_label, dt=1 / 30):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array(
            [
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
            ]
        )
        self.kf.R = R
        self.kf.P = np.diag([1, vmax**2 / 3.0, 1, vmax**2 / 3.0])

        G = np.array(
            [
                [0.5 * dt**2, 0],
                [dt, 0],
                [0, 0.5 * dt**2],
                [0, dt],
            ]
        )
        Q0 = np.diag([wx, wy])
        self.kf.Q = G @ Q0 @ G.T

        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = y[1]
        self.kf.x[3] = 0

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
        self.kf.update(y, R)

    def predict(self):
        """Predict the next state."""
        self.kf.predict()
        self.age += 1
        return self.kf.H @ self.kf.x

    def get_state(self):
        """Get the current state estimate."""
        return self.kf.x

    def distance(self, y, R):
        """Compute the Mahalanobis distance between predicted and observed states."""
        diff = y - (self.kf.H @ self.kf.x)
        S = self.kf.H @ self.kf.P @ self.kf.H.T + R
        SI = np.linalg.inv(S)
        mahalanobis = diff.T @ SI @ diff
        logdet = np.log(np.linalg.det(S))
        return mahalanobis[0, 0] + logdet
