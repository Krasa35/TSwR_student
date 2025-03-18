import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)
        self.Kd = -1
        self.Kp = -1

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        #--------------------1-------------------------
        # v = q_r_ddot  
        #--------------------2-------------------------
        v = self.model.M(x) @ q_r_ddot + self.model.C(x) @ q_r_dot
        #--------------------3-------------------------
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])
        v = v + self.Kd*(q_dot - q_r_dot) + self.Kp*(q - q_r)
        return v
