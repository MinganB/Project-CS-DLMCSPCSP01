import numpy as np

class differential_solver:
    """
    Base class for solving ordinary differential equations.
    """
    def __init__(self, f):
        self.f = f

    def step(self):
        """
        Move one time step forward.
        This method must be implemented in the subclass.
        """
        raise NotImplementedError
    
    def set_initial(self, U0):
        """
        Set the initial condition.
        U0 can be a scalar or a numpy array.
        neq is the number of equations in the system.
        """
        if isinstance(U0, (float, int)):
            self.neq = 1
            U0 = float(U0)
        else:
            U0 = np.asarray(U0)
            if U0.ndim == 0:
                self.neq = 1
            else:
                self.neq = U0.size
        self.U0 = U0

    def solve(self, time_points):
        """
        Find u for all time points.
        u is an array where u[i] is the solution at time t[i].
        """
        if not isinstance(time_points, (list, np.ndarray)):
            raise ValueError('time_points should be a list or array')

        self.t = np.asarray(time_points)
        n = self.t.size

        self.u = np.zeros((n, self.neq))
        self.u[0, :] = self.U0

        for i in range(n-1):
            self.i = i
            self.u[i+1, :] = self.step()

        return self.u[:i+2], self.t[:i+2]

class ForwardEuler(differential_solver):
    """
    Forward Euler method for solving ordinary differential equations.
    Step forward in time using the formula:
    u[i+1] = u[i] + dt*f(u[i], t[i])
    """

    def __init__(self, f):
        differential_solver.__init__(self, f)

    def step(self):
        u, f, i, t = self.u, self.f, self.i, self.t
        dt = t[i+1] - t[i]
        unew = u[i, :] + dt*f(u[i, :], t[i])
        return unew