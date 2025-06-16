import numpy as np

# IDEAL
# self.l1 = 0.5
# self.r1 = 0.04
# self.m1 = 3.0
# self.l2 = 0.4
# self.r2 = 0.04
# self.m2 = 2.4
# self.m3 = 2.0
# self.r3 = 0.05

class ManipulatorModelMMAC:
    def __init__(self, Tp, m3, r3):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.0
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        self.d1 = self.l1/2
        self.d2 = self.l2/2


    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        #   Without the object at the tip
        alpha   = self.m1*self.d1**2+self.I_1+self.m2*(self.l1**2+self.d2**2)+self.I_2
        beta    = self.m2*self.l1*self.d2
        gamma   = self.m2*self.d2**2 + self.I_2   
        #   With the object at the tip
        alpha = self.I_1 + self.I_2 + self.m1 * self.d1 ** 2 + self.m2 * (self.l1 ** 2 + self.d2 ** 2) + \
                self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        beta = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        gamma = self.I_2 + self.m2 * self.d2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2

        m11 = alpha + 2*beta*np.cos(q2)
        m12 = gamma + beta*np.cos(q2)
        m21 = gamma + beta*np.cos(q2)
        m22 = gamma

        return np.array([[m11, m12],[m22, m21]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """

        q1, q2, q1_dot, q2_dot = x
        #   Without the object at the tip
        beta    = self.m2*self.l1*self.d2
        #   With the object at the tip
        beta = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2

        c11 = -beta*np.sin(q2)*q2_dot
        c12 = -beta*np.sin(q2)*(q1_dot+q2_dot)
        c21 = beta*np.sin(q1)*q1_dot
        c22 = 0
        return np.array([[c11, c12],[c21, c22]])
