import numpy as np

class Est:
    def __init__(self, pos, ax, dt):
        self.EstState = pos
        self.drift_noise_GPS = np.zeros(2)
        self.drift_noise_Dyn_Eng = np.zeros(2)
        self.drift_noise_Dyn_Cur = np.zeros(2)
        self.gps_lastError = np.zeros(2)

        self.R = 64 * np.eye(2)

        sig_a = 1.25
        mat = np.array([[dt**4/4, 0, dt**3/2, 0],
                        [0, dt**4/4, 0, dt**3/2],
                        [dt**3/2, 0, dt**2, 0],
                        [0, dt**3/2, 0,  dt**2]])

        self.Q = sig_a**2 * mat

        self.current_GPS = np.zeros(2)
        self.current_Dyn = np.zeros(4)

        self.P = np.diag([64.0, 64.0, 100.0, 100.0])

        self.A = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]
                          )
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        
        self.Hv = np.array([[0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

        self.Rv = 34 * np.eye(2)
        self.current_Vmeas = np.zeros(2)


    def getGPS(self, Boat_State, pos, t):
        jitter_noise = np.random.normal(0, 8, size = 2)
        self.drift_noise_GPS += np.random.normal(0, 0.1, size = 2)

        GPS = Boat_State + jitter_noise + self.drift_noise_GPS

        self.current_GPS = GPS

        pos.append([self.current_GPS[0], self.current_GPS[1], t])


    def dynamic_predict(self):
        self.current_Dyn = self.A @ self.EstState

        self.P = self.A @ self.P @ self.A.T + self.Q
    

    def update_vel(self):
        y = self.current_Vmeas - self.Hv @ self.current_Dyn

        S = self.Hv @ self.P @ self.Hv.T + self.Rv

        K = self.P @ self.Hv.T @ np.linalg.inv(S)

        self.EstState = self.current_Dyn + K @ y

        self.P = (np.eye(4) - K @ self.Hv) @ self.P


    def get_Vmeas(self, current, eng_V, pos, t, dt):
        current_noise = np.random.normal(0, 10, size = 2)
        eng_noise = np.random.normal(0, 8, size = 2)
        self.drift_noise_Dyn_Eng += np.random.normal(0, 0.1, size = 2)
        self.drift_noise_Dyn_Cur += np.random.normal(0, 0.1, size = 2)

        curr_N = current + current_noise + self.drift_noise_Dyn_Cur
        eng_N = eng_V + eng_noise + self.drift_noise_Dyn_Eng

        self.current_Vmeas = curr_N + eng_N

        dynamic_inp_guess = self.EstState[0:2] + self.current_Vmeas * dt

        pos.append([dynamic_inp_guess[0], dynamic_inp_guess[1], t])

    
    def update_est(self, pos, t):
        y = self.current_GPS - self.H @ self.EstState # suprise term

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.EstState = self.EstState + K @ y

        pos.append([float(self.EstState[0]), float(self.EstState[1]), t])

        self.P = (np.eye(4) - K @ self.H) @ self.P
    

    def dyn_only_update(self, t, est_pos):
        self.current_Dyn = self.A @ self.EstState 

        self.EstState = self.current_Dyn

        est_pos.append([float(self.EstState[0]), float(self.EstState[1]), t])

        self.P = self.A @ self.P @ self.A.T + self.Q
