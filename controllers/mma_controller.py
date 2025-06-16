import numpy as np
from .controller import Controller
from models.manipulator_model_mmac import ManipulatorModelMMAC
from controllers.feedback_linearization_controller import FeedbackLinearizationController


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        controller = FeedbackLinearizationController(Tp)
        self.models = [ManipulatorModelMMAC(Tp, 0.1, 0.05), 
                       ManipulatorModelMMAC(Tp, 0.01, 0.01), 
                       ManipulatorModelMMAC(Tp, 1.0, 0.3)]
        self.i = 0
        self.prev_u = np.zeros((2, 1))
        self.Kd = -1.2
        self.Kp = -1.7

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q = x[:2]     
        q_dot = x[2:] 
        errors = []
        for model in self.models:
            M = model.M(x)
            C = model.C(x)
            q_ddot = np.linalg.inv(M) @ (self.prev_u.flatten() - C @ q_dot)
            q_next = q + model.Tp * q_dot + 0.5 * model.Tp**2 * q_ddot
            q_dot_next = q_dot + model.Tp * q_ddot
            x_pred = np.concatenate([q_next, q_dot_next])
            error = np.linalg.norm(x - x_pred)
            errors.append(error)

        self.i = np.argmin(errors)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        # v = q_r_ddot # TODO: add feedback
        v = q_r_ddot + self.Kd*(q_dot - q_r_dot) + self.Kp*(q - q_r) # ADDED feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prev_u = u
        return u
