import numpy as np
from lap import lapjv
import line_profiler
from numba import njit
from .kalman import KalmanTracker, TrackStatus


def linear_assignment(cost_matrix, thresh):
    """Solve the linear assignment problem with a threshold."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


class UCMCTrack:
    """Unified Camera Motion Compensation Tracker with class integration."""

    def __init__(
        self, a1, a2, wx, wy, vmax, max_age, fps, dataset, high_score, use_cmc, detector=None
    ):
        self.wx = wx
        self.wy = wy
        self.vmax = vmax
        self.dataset = dataset
        self.high_score = high_score
        self.max_age = max_age
        self.a1 = a1
        self.a2 = a2
        self.dt = 1.0 / fps
        self.use_cmc = use_cmc

        self.trackers = []
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []

        self.detector = detector
    
    def update(self, dets, frame_id):
        """Update the tracker with new detections."""
        self.data_association(dets, frame_id)
        self.associate_tentative(dets)
        self.initial_tentative(dets)
        self.delete_old_trackers()
        self.update_status(dets)

    def data_association(self, dets, frame_id):
        """Associate detections with existing tracks."""
        # Separate detections into high score and low score
        detidx_high = [i for i, det in enumerate(dets) if det.conf >= self.high_score]
        detidx_low = [i for i, det in enumerate(dets) if det.conf < self.high_score]

        # Predict new locations of tracks
        for track in self.trackers:
            track.predict()
            if self.use_cmc:
                x, y = self.detector.cmc(
                    track.kf.x[0, 0], track.kf.x[2, 0], track.w, track.h, frame_id
                )
                track.kf.x[0, 0] = x
                track.kf.x[2, 0] = y

        trackidx_remain = []
        self.detidx_remain = []

        # Combine confirmed and coasted indices
        trackidx = self.confirmed_idx + self.coasted_idx

        # Vectorized association for high score detections
        self._associate_detections_to_tracks(
            dets, detidx_high, trackidx, self.a1, is_high_score=True
        )

        # Vectorized association for low score detections
        self._associate_detections_to_tracks(
            dets, detidx_low, trackidx_remain, self.a2, is_high_score=False
        )
    @line_profiler.profile
    def _associate_detections_to_tracks(self, dets, det_indices, track_indices, threshold, is_high_score):
        """Associate detections to tracks using vectorized operations."""
        if not det_indices or not track_indices:
            if is_high_score:
                self.detidx_remain.extend(det_indices)
            else:
                for idx in track_indices:
                    track = self.trackers[idx]
                    track.status = TrackStatus.Coasted
                    track.detidx = -1
            return

        # Prepare detection observations and covariances
        det_observations = np.array([dets[idx].y.flatten() for idx in det_indices])
        det_covariances = np.array([dets[idx].R for idx in det_indices])

        # Prepare track predictions and covariances
        track_predictions = np.array([self.trackers[idx].kf.H @ self.trackers[idx].kf.x for idx in track_indices])
        track_covariances = np.array([self.trackers[idx].kf.H @ self.trackers[idx].kf.P @ self.trackers[idx].kf.H.T for idx in track_indices])

        # Compute cost matrix
        num_dets = len(det_indices)
        num_tracks = len(track_indices)
        cost_matrix = np.zeros((num_dets, num_tracks))

        for i in range(num_dets):
            y = det_observations[i]
            R = det_covariances[i]
            diff = y - track_predictions.reshape(num_tracks, -1)
            S = track_covariances + R
            SI = np.linalg.inv(S)
            mahalanobis = np.einsum('ij,ijk,ik->i', diff, SI, diff)
            logdet = np.log(np.linalg.det(S))
            cost_matrix[i] = mahalanobis + logdet

        matched_indices, unmatched_dets, unmatched_tracks = linear_assignment(cost_matrix, threshold)

        # Process matched detections and tracks
        for i, j in matched_indices:
            det_idx = det_indices[i]
            trk_idx = track_indices[j]
            track = self.trackers[trk_idx]
            track.update(dets[det_idx].y, dets[det_idx].R)
            track.death_count = 0
            track.detidx = det_idx
            track.status = TrackStatus.Confirmed
            dets[det_idx].track_id = track.id

        # Update unmatched detections and tracks
        if is_high_score:
            self.detidx_remain.extend(det_indices[i] for i in unmatched_dets)
        else:
            for i in unmatched_tracks:
                trk_idx = track_indices[i]
                track = self.trackers[trk_idx]
                track.status = TrackStatus.Coasted
                track.detidx = -1

    def associate_tentative(self, dets):
        """Associate remaining detections with tentative tracks."""
        if not self.detidx_remain or not self.tentative_idx:
            return

        # Prepare detection observations and covariances
        det_observations = np.array([dets[idx].y.flatten() for idx in self.detidx_remain])
        det_covariances = np.array([dets[idx].R for idx in self.detidx_remain])

        # Prepare track predictions and covariances
        track_predictions = np.array([self.trackers[idx].kf.H @ self.trackers[idx].kf.x for idx in self.tentative_idx])
        track_covariances = np.array([self.trackers[idx].kf.H @ self.trackers[idx].kf.P @ self.trackers[idx].kf.H.T for idx in self.tentative_idx])

        # Compute cost matrix
        num_dets = len(self.detidx_remain)
        num_tracks = len(self.tentative_idx)
        cost_matrix = np.zeros((num_dets, num_tracks))

        for i in range(num_dets):
            y = det_observations[i]
            R = det_covariances[i]
            diff = y - track_predictions.reshape(num_tracks, -1)
            S = track_covariances + R
            SI = np.linalg.inv(S)
            mahalanobis = np.einsum('ij,ijk,ik->i', diff, SI, diff)
            logdet = np.log(np.linalg.det(S))
            cost_matrix[i] = mahalanobis + logdet

        matched_indices, unmatched_dets, unmatched_tracks = linear_assignment(cost_matrix, self.a1)

        # Process matched detections and tracks
        for i, j in matched_indices:
            det_idx = self.detidx_remain[i]
            trk_idx = self.tentative_idx[j]
            track = self.trackers[trk_idx]
            track.update(dets[det_idx].y, dets[det_idx].R)
            track.death_count = 0
            track.birth_count += 1
            track.detidx = det_idx
            dets[det_idx].track_id = track.id
            if track.birth_count >= 2:
                track.birth_count = 0
                track.status = TrackStatus.Confirmed

        # Increment death count for unmatched tracks
        for i in unmatched_tracks:
            trk_idx = self.tentative_idx[i]
            track = self.trackers[trk_idx]
            track.death_count += 1
            track.detidx = -1

        # Update unmatched detections
        self.detidx_remain = [self.detidx_remain[i] for i in unmatched_dets]

    def initial_tentative(self, dets):
        """Initialize new tentative tracks from unmatched detections."""
        for idx in self.detidx_remain:
            det = dets[idx]
            new_tracker = KalmanTracker(
                det.y,
                det.R,
                self.wx,
                self.wy,
                self.vmax,
                det.bb_width,
                det.bb_height,
                det.det_class,
                self.dt,
            )
            new_tracker.status = TrackStatus.Tentative
            new_tracker.detidx = idx
            self.trackers.append(new_tracker)
        self.detidx_remain = []

    def delete_old_trackers(self):
        """Delete tracks that have not been updated."""
        self.trackers = [
            trk
            for trk in self.trackers
            if not (
                (trk.status == TrackStatus.Coasted and trk.death_count >= self.max_age)
                or (trk.status == TrackStatus.Tentative and trk.death_count >= 2)
            )
        ]

    def update_status(self, dets):
        """Update the status lists of tracks."""
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []

        for i, trk in enumerate(self.trackers):
            detidx = trk.detidx
            if 0 <= detidx < len(dets):
                trk.h = dets[detidx].bb_height
                trk.w = dets[detidx].bb_width

            if trk.status == TrackStatus.Confirmed:
                self.confirmed_idx.append(i)
            elif trk.status == TrackStatus.Coasted:
                self.coasted_idx.append(i)
            elif trk.status == TrackStatus.Tentative:
                self.tentative_idx.append(i)
