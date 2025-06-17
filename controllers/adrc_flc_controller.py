import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManipulatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L =  np.array([[3*p[0], 0],
                           [0, 3*p[1]],
                           [3*p[0]**2, 0],
                           [0, 3*p[1]**2],
                           [p[0]**3, 0],
                           [0, p[1]**3]])
        W = np.array([[1., 0., 0., 0., 0., 0.],[0., 1., 0., 0., 0., 0.]])
        A = np.array([[0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.]])
        B = np.zeros((6,2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)
        C = self.model.C(x)
        A = np.zeros((6, 6))
        A[0, 2] = 1
        A[1, 3] = 1
        A[2, 4] = 1
        A[3, 5] = 1
        A[2:4, 2:4] = -M_inv @ C
        B = np.zeros((6, 2))
        B[2:4, :] = M_inv
        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        M = self.model.M(x)
        C = self.model.C(x)
        z_hat = self.eso.get_state()
        x_hat = z_hat[0:2]
        x_hat_dot = z_hat[2:4]
        f = z_hat[4:]
        v = q_d_ddot + self.Kd @ (q_d_dot - x_hat_dot) + self.Kp @ (q_d - q)
        u = M @ (v - f) + C @ x_hat_dot
        self.update_params(x_hat, x_hat_dot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))
        return u