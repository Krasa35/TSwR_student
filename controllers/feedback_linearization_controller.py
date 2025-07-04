import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)
        self.Kd = -1
        self.Kp = -1.5

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q_dot = x[2:]
        q = x[:2]
        #--------------------1-------------------------
        # v = q_r_ddot  
        #--------------------2-------------------------
        v = q_r_ddot + self.Kd*(q_dot - q_r_dot) + self.Kp*(q - q_r)
        #--------------------3-------------------------
        v = self.model.M(x) @ v + self.model.C(x) @ q_dot
        return v
