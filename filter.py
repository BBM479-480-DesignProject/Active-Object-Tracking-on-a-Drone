from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise
import numpy as np

class PointEKF:
    def __init__(self, dt, process_noise_var=0.1, measurement_noise_var=5.0):
        self.dt = dt
        self.ekf = EKF(dim_x=4, dim_z=2)  # State includes x, y, vx, vy

        # Initial state estimate (x, y, vx, vy)
        self.ekf.x = np.array([0., 0., 1., 1.])  # Example initial state

        # State covariance matrix
        self.ekf.P = np.eye(4) * 500.

        # Process noise covariance
        self.ekf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_var, block_size=2)

        # Measurement noise covariance
        self.ekf.R = np.eye(2) * measurement_noise_var

    def state_transition_function(self, x):
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        return F @ x

    def measurement_function(self, x):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        return H @ x

    def jacobian_F(self, x):
        return np.array([[1, 0, self.dt, 0],
                         [0, 1, 0, self.dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def jacobian_H(self, x):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    def predict(self):
        self.ekf.F = self.jacobian_F(self.ekf.x)
        self.ekf.predict()

    def update(self, z):
        # Ensure the measurement vector z is a numpy array
        z = np.array(z)
        # Ensure the measurement vector z is a column vector
        if z.ndim == 1:
            z = z.reshape(-1, 1)

        # Compute the innovation (measurement residual)
        Hx = self.measurement_function(self.ekf.x)
        self.ekf.y = z - Hx.reshape(-1, 1)

        # Compute the Kalman gain
        H = self.jacobian_H(self.ekf.x)
        S = H @ self.ekf.P @ H.T + self.ekf.R
        self.ekf.K = self.ekf.P @ H.T @ np.linalg.inv(S)

        # Update the state estimate
        self.ekf.x = self.ekf.x + (self.ekf.K @ self.ekf.y).flatten()

        # Update the covariance matrix
        I = np.eye(self.ekf.dim_x)
        self.ekf.P = (I - self.ekf.K @ H) @ self.ekf.P

    def get_state(self):
        return self.ekf.x

    def get_covariance(self):
        return self.ekf.P

class PointTrackerKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.kf = KalmanFilter(dim_x=2, dim_z=2)
        self.kf.x = initial_state
        self.kf.P = initial_covariance
        self.kf.Q = process_noise
        self.kf.R = measurement_noise
        self.kf.H = np.eye(2)
        self.kf.F = np.eye(2)

    def predict(self):
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(measurement)

    def get_predicted_state(self):
        return self.kf.x  # Extract position from state vector


class BoundingBoxKalmanFilter:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State vector [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.kf.x = np.zeros(8)

        # State transition matrix
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])

        # Measurement uncertainty
        self.kf.R *= 10

        # Process uncertainty
        self.kf.Q = np.eye(8)

        # Initial covariance matrix
        self.kf.P *= 100

    def update(self, x1, y1, x2, y2):
        z = np.array([x1, y1, x2, y2])
        self.kf.update(z)

    def predict(self):
        self.kf.predict()

    def get_state(self):
        x1, y1, x2, y2 = self.kf.x[:4]
        return x1, y1, x2, y2