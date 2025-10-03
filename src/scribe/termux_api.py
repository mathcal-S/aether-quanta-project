import subprocess
import json
import numpy as np
from scipy.linalg import inv
from datetime import datetime
import requests

class TermuxAPI:
    def __init__(self):
        self.state = np.zeros(12)  # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, ori_x, ori_y, ori_z, bias_gx, bias_gy, bias_gz]
        self.P = np.eye(12) * 0.1  # Covariance
        self.Q = np.eye(12) * 0.01  # Process noise
        self.R = np.eye(9) * 0.1  # Measurement noise
        self.phi = (1 + np.sqrt(5)) / 2
        self.rnn_weights = np.random.rand(12, 12)  # Mock RNN for prediction
        self.rnn_h = np.zeros(12)  # RNN hidden state

    def run_termux_command(self, cmd_args):
        try:
            result = subprocess.check_output(cmd_args, stderr=subprocess.STDOUT)
            text = result.decode('utf-8').strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        except Exception as e:
            print(f"[Termux API Error] Command {' '.join(cmd_args)} failed: {e}")
            return None

    def get_all_sensors(self):
        return self.run_termux_command(['termux-sensor', '-l'])

    def get_sensor_data(self, sensor_name):
        return self.run_termux_command(['termux-sensor', '-s', sensor_name, '-n', '1'])

    def fuse_sensors(self):
        # Fetch sensors
        accel = self.get_sensor_data("accelerometer") or {"values": [0, 0, 9.81]}
        gyro = self.get_sensor_data("gyroscope") or {"values": [0, 0, 0]}
        mag = self.get_sensor_data("magnetic_field") or {"values": [0, 0, 50]}
        light = self.get_sensor_data("light") or {"values": [400]}
        prox = self.get_sensor_data("proximity") or {"values": [0]}
        baro = self.get_sensor_data("pressure") or {"values": [1013.25]}

        # Dynamic noise tuning (variance-based)
        sensor_vars = [np.var(accel["values"]), np.var(gyro["values"]), np.var(mag["values"])]
        self.R = np.eye(9) * (0.1 + np.mean(sensor_vars) * 0.05)

        # RNN prediction for state transition (hybrid EKF-RNN)
        self.rnn_h = np.tanh(self.rnn_weights @ self.rnn_h + self.state)
        self.state = self.rnn_h  # RNN forecast

        # EKF prediction
        F = np.eye(12)
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

        # Measurement update
        z = np.array(accel["values"] + gyro["values"] + mag["values"])
        H = np.eye(9, 12)
        y = z.T - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P

        # ESQET D_ent from fused state variance
        d_ent = np.var(self.state[:6])
        fqc = (1 + self.phi * np.pi * 0.5 * d_ent) * np.cos(0.5)
        return {"state": self.state.tolist(), "covariance": self.P.diagonal().tolist(), "fqc": fqc}

    # Other methods (take_photo, record_audio, etc.) unchanged from prior

if __name__ == "__main__":
    api = TermuxAPI()
    for i in range(5):
        fused = api.fuse_sensors()
        print(f"Fused State: {fused['state'][:3]}, F_QC: {fused['fqc']:.4f}")
        time.sleep(1)
