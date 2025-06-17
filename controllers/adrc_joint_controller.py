import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [b],
                      [0]])
        L = np.array([[3*p],
                      [3*p**2],
                      [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)
        self.prev_u = 0.

    def set_b(self, b):
        self.b = b
        B = np.array([[0.], [self.b], [0.]])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q_hat, q_hat_dot, f = self.eso.get_state()
        v = q_d_ddot + self.kp * (q_d - q_hat) +  self.kd * (q_d_dot - q_hat_dot)
        self.b = np.linalg.inv(self)
        u = (v - f) / self.b
        self.last_u = u
        self.eso.update(x[0], self.last_u)
        return u
