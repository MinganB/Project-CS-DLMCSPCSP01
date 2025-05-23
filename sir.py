"""
SIR false news spread model using the Forward Euler method.
Definition of the equations:

S = susceptible population (people who can be influenced to become spreaders)
I = influenced population (spreaders of misinformation)
R = recovered population (people who have recovered from the misinformation)

S' = -beta * S * I
I' = beta * S * I - mu * I
R' = mu * I

where:
    beta = influence rate from S to I
    mu = transition rate from I to R
"""

import numpy as np
from differential_solver import ForwardEuler
from matplotlib import pyplot as plt
from guizero import App, Text, TextBox, PushButton

class SIR:
    def __init__(self, beta, mu, S0, I0, R0):
        """
        S0, I0, R0: initial values of S, I, R
        beta: influence rate from S to I
        mu: transition rate from I to R
        """

        if isinstance(mu, (float, int)):
            self.mu = lambda t: mu      # treat mu as a constant
        elif callable(mu):
            self.mu = mu                # treat mu as a function

        if isinstance(beta, (float, int)):
            self.beta = lambda t: beta  # treat beta as a constant
        elif callable(beta):
            self.beta = beta            # treat beta as a function

        self.initial_conditions = [S0, I0, R0]

    def __call__(self, u, t):
        """
        Define the differential equations.
        u = [S, I, R]
        t = time
        """
        S, I, _ = u
        beta = self.beta(t)
        mu = self.mu(t)

        if(t == 0):
            print(f"S: {S}, I: {I}, beta: {beta}, mu: {mu}")

        susceptible = -beta * S * I
        influenced = beta * S * I - mu * I
        recovered = mu * I

        return np.asarray([susceptible, influenced, recovered])

def visualize_sir_model(u, t):
    """
    Visualize the SIR model outcome.
    """
    plt.figure()
    plt.plot(t, u[:, 0], label='Susceptible')
    plt.plot(t, u[:, 1], label='Influenced')
    plt.plot(t, u[:, 2], label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model of Misinformation Spread')
    plt.legend()

    plt.show()

if __name__ == "__main__":

    enable_gui = False # Set to True to enable GUI for parameter inputs

    def run_model():
        """
        Run the SIR model with the given parameters.
        """
        if enable_gui:
            beta_t0 = float(beta_t0_box.value)/1000
            intervention_day = int(intervention_day_box.value)
            intervention_beta = float(intervention_beta_box.value)/1000
            mu_t0 = float(mu_t0_box.value)/10
            intervention_mu = float(intervention_mu_box.value)/10
            S0 = int(S0_box.value)
            I0 = int(I0_box.value)
            R0 = int(R0_box.value)
            n_days = int(n_days_box.value)
            resolution = int(resolution_box.value)
        else:
            """
            The default parameters for the SIR model, obtained from the data_analysis.py module.
            """
            beta_t0 = 0.7701/1000
            intervention_day = 20
            intervention_beta = 0.4995/1000
            mu_t0 = 0.2233/10
            intervention_mu = 0.2233/10
            S0 = 1000
            I0 = 1
            R0 = 0
            n_days = 140
            resolution = 1001

        # Use lambda functions to define beta and mu as functions of time
        beta = lambda t: beta_t0 if t < intervention_day else intervention_beta
        mu = lambda t: mu_t0 if t < intervention_day else intervention_mu

        sir = SIR(beta=beta, mu=mu, S0=S0, I0=I0, R0=R0)
        solver = ForwardEuler(sir)
        solver.set_initial(sir.initial_conditions)

        time_points = np.linspace(0, n_days, resolution)
        u, t = solver.solve(time_points)

        visualize_sir_model(u, t)

    if enable_gui:
        """
        Create a GUI for input of model parameters.
        """
        app = App(title="SIR Model - Parameter Input")

        text = Text(app, text="SIR Misinformation Model", size=20)
        Text(app, text="Use the text boxes to adjust the parameters of the SIR model.")

        Text(app, text="-" * 50)

        Text(app, text="Initial influence rate (beta):") # beta_t0
        beta_t0_box = TextBox(app, text="0.7701")

        Text(app, text="Initial recovery rate (mu):") # mu_t0
        mu_t0_box = TextBox(app, text="0.2233")

        Text(app, text="-" * 50)

        Text(app, text="Intervention day:")
        intervention_day_box = TextBox(app, text="20")

        Text(app, text="Adjusted influence rate (beta after intervention):")
        intervention_beta_box = TextBox(app, text="0.4995")

        Text(app, text="Adjusted recovery rate (mu after intervention):")
        intervention_mu_box = TextBox(app, text="0.2233")

        Text(app, text="-" * 50)

        Text(app, text="Initial susceptible population:") # S0
        S0_box = TextBox(app, text="1000")

        Text(app, text="Initial influenced population:") # I0
        I0_box = TextBox(app, text="1")

        Text(app, text="Initial recovered population:") # R0
        R0_box = TextBox(app, text="0")

        Text(app, text="-" * 50)

        Text(app, text="Number of days:")
        n_days_box = TextBox(app, text="140")

        Text(app, text="Time resolution (number of points):")
        resolution_box = TextBox(app, text="1001")

        Text(app, text="-" * 50)

        PushButton(app, text="Run Model", command=run_model)

        app.height = 10000
        app.display()
    else:
        run_model()