import numpy as np
from trajectory_generators.trajectory_generator import TrajectoryGenerator


class Poly3(TrajectoryGenerator):
    def __init__(self, start_q, desired_q, T):
        self.T = T
        self.q_0 = start_q
        self.q_k = desired_q
        """
        Please implement the formulas for a_0 till a_3 using self.q_0 and self.q_k
        Assume that the velocities at start and end are zero.
        """
        """
        Used paper to come to solution:
        q_r(0)      = a_0           = q_0
        q_dot_r(0)  = a_1 + 3 * a_0 = q_dot_0
        q_r(1)      = a_3           = q_desired
        q_dot_r(1)  = 3 * a_3 - a_2 = q_dot_desired
        Below implemented without usage of initial velocity and desired velocity -
         awaiting to be stopped at the begginning and at the end
        """

        self.a_0 = self.q_0
        self.a_1 = - 3 * self.q_0 + start_q
        self.a_2 = 3 * self.q_k  - desired_q
        self.a_3 = self.q_k

    def generate(self, t):
        """
        Implement trajectory generator for your manipulator.
        Positional trajectory should be a 3rd degree polynomial going from an initial state q_0 to desired state q_k.
        Remember to derive the first and second derivative of it also.
        Use following formula for the polynomial from the instruction.
        """
        """
        Used Wolfram to calculate derivatives from q
        """
        t /= self.T
        q = self.a_3 * t**3 + self.a_2 * t**2 * (1 - t) + self.a_1 * t * (1 - t)**2 + self.a_0 * (1 - t)**3
        q_dot = 3 * self.a_3 * t**2 + self.a_2 * (2 * t - 3 * t**2) + \
                self.a_1 * (3 * t**2 - 4 * t + 1) + 3 * self.a_0 * (1 - t)**2
        q_ddot = 6 * self.a_3 * t + self.a_2 * (2 - 6 * t) + self.a_1 * (6 * t - 4) + 6 * self.a_0 * (1 - t)
        return q, q_dot / self.T, q_ddot / self.T**2
