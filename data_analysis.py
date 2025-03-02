import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def load_dataset():
    df = pd.read_csv('data/covid-misinfo-videos.csv')

    df = df.dropna(subset=['twitter_post_ids', 'facebook_post_ids', 'removal_timestamp', 'published_timestamp'])

    df['twitter_post_count'] = df['twitter_post_ids'].apply(lambda x: len(eval(x)))
    df.drop(columns=['twitter_post_ids'], inplace=True)

    df['facebook_post_count'] = df['facebook_post_ids'].apply(lambda x: len(eval(x)))
    df.drop(columns=['facebook_post_ids'], inplace=True)

    # time taken to remove video
    df['removal_timestamp'] = pd.to_datetime(df['removal_timestamp'])
    df['published_timestamp'] = pd.to_datetime(df['published_timestamp'])

    df['days_to_remove'] = (df['removal_timestamp'] - df['published_timestamp']).dt.days
    df = df[df['days_to_remove'] >= 0]  # Filter rows with incorrect negative days_to_remove

    df['total_engagement'] = df['facebook_post_count'] + df['twitter_post_count'] + df['facebook_graph_reactions'] + df['facebook_graph_comments'] + df['facebook_graph_shares']
    df['engagement_per_day'] = df['total_engagement'] / df['days_to_remove'].replace(0, 1) # Replace 0 with 1 to avoid division by zero

    # drop columns not needed
    df.drop(columns=['archive_url', 'channel_id', 'removal_timestamp'], inplace=True)

    print(df.head())
    print('Number of videos:', len(df))
    print(df.columns)

    return df

def get_daily_engagements(df):
    max_days = df['days_to_remove'].max()
    daily_engagements = df.groupby('days_to_remove')['total_engagement'].sum().reindex(range(1, max_days + 1), fill_value=0)

    return pd.DataFrame({'day': daily_engagements.index, 'engagement_sum': daily_engagements.values})

def get_daily_cumulative_engagements(df):
    max_days = df['days_to_remove'].max()
    engagement_sums = []

    for day in range(1, max_days + 1):
        engagement_sum = df[df['days_to_remove'] < day]['total_engagement'].sum()
        engagement_sums.append({'day': day, 'engagement_sum': engagement_sum})

    return pd.DataFrame(engagement_sums)

def plot_engagements(daily_engagements, cumulative_engagements):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # plot daily engagements
    axs[0].plot(daily_engagements['day'], daily_engagements['engagement_sum'] / 1e6)
    axs[0].set_xlabel('Days since video published')
    axs[0].set_ylabel('Number of new engagements (millions)')
    axs[0].set_title('Number of new engagements per day since video published')

    # plot cumulative engagements
    axs[1].plot(cumulative_engagements['day'], cumulative_engagements['engagement_sum'] / 1e6)
    axs[1].set_xlabel('Days since video published')
    axs[1].set_ylabel('Total engagements (millions)')
    axs[1].set_title('Total users engagements since video publication')

    plt.tight_layout()
    plt.title('Engagements over time')
    plt.show()

def get_infected_recovered(df, day):
    '''
    Calculate the total cumulative number of infected and recovered users at a given day.
    This calculation assumes that the engagement rate is constant over time, which might not be the case in reality.
    Incorporation of time-series data would allow a more sophisticated model.
    '''
    infected = df[df['days_to_remove'] > day].apply(lambda row: day * row['engagement_per_day'], axis=1).sum()
    recovered = df[df['days_to_remove'] <= day]['total_engagement'].sum()

    susceptible = df['total_engagement'].sum() - infected - recovered

    return int(round(infected)), int(round(recovered)), int(round(susceptible))

def plot_infection_statistics(infected_recovered_data):
    infected_recovered_df = pd.DataFrame(infected_recovered_data)
    plt.plot(infected_recovered_df['day'], infected_recovered_df['susceptible'] / 1e6, label='Susceptible', color='blue')
    plt.plot(infected_recovered_df['day'], infected_recovered_df['infected'] / 1e6, label='Infected', color='orange')
    plt.plot(infected_recovered_df['day'], infected_recovered_df['recovered'] / 1e6, label='Recovered', color='green')
    plt.xlabel('Days since video published')
    plt.ylabel('Number of users (millions)')
    plt.title('Number of infected, recovered, and susceptible users over time')
    plt.legend()
    plt.show()

def estimate_sir_parameters(infected_recovered_data_df):
    """
    Estimate the parameters beta (infection rate) and mu (recovery rate) for the SIR model.

    Parameters:
    infected_recovered_data_df (pd.DataFrame): A DataFrame with columns 'day', 'infected', 'recovered', 'susceptible'.

    Returns:
    tuple: (beta, mu) - Estimated parameters.
    """
    # Extract the data
    S = infected_recovered_data_df['susceptible'].values
    I = infected_recovered_data_df['infected'].values
    R = infected_recovered_data_df['recovered'].values
    N = S[0] + I[0] + R[0]  # Total population (assumed constant)

    # Compute the derivatives using finite differences
    dSdt = np.diff(S)  # dS/dt ≈ S(t+1) - S(t)
    dIdt = np.diff(I)  # dI/dt ≈ I(t+1) - I(t)
    dRdt = np.diff(R)  # dR/dt ≈ R(t+1) - R(t)

    # Define the model for the derivatives
    def sir_derivatives(t, beta, mu):
        S_t, I_t, _ = t
        dSdt_model = -beta * (S_t * I_t / N)
        dIdt_model = beta * (S_t * I_t / N) - mu * I_t
        dRdt_model = mu * I_t
        return np.concatenate((dSdt_model, dIdt_model, dRdt_model))

    # Prepare the data for fitting
    # We exclude the last day because we are computing differences
    SIR_data = np.vstack((S[:-1], I[:-1], R[:-1])).T  # Input to the model
    observed_derivatives = np.concatenate((dSdt, dIdt, dRdt))  # Target for fitting

    # Initial guesses for beta and mu
    initial_guess = [0.001, 0.1]

    # Fit the model using curve_fit
    params, _ = curve_fit(
        sir_derivatives,
        SIR_data.T,
        observed_derivatives,
        p0=initial_guess,
        maxfev=10000  # Increase max iterations if needed
    )

    # Extract the estimated parameters
    beta, mu = np.round(params, 4)
    return beta, mu

if __name__ == '__main__':
    dataset = load_dataset()
    dataset.to_csv('data/covid-misinfo-videos-cleaned.csv', index=False)

    daily_engagements = get_daily_engagements(dataset)
    cumulative_engagements = get_daily_cumulative_engagements(dataset)

    print(cumulative_engagements.head())

    # plot daily engagements and cumulative engagements side by side
    plot_engagements(daily_engagements, cumulative_engagements)

    infected_recovered_data = []

    for day in range(91):
        infected, recovered, susceptible = get_infected_recovered(dataset, day)
        infected_recovered_data.append({'day': day, 'infected': infected, 'recovered': recovered, 'susceptible': susceptible})

    print("Initial population: ", infected_recovered_data[1])

    # plot infected, recovered, and susceptible users
    plot_infection_statistics(infected_recovered_data)

    beta, mu = estimate_sir_parameters(pd.DataFrame(infected_recovered_data))
    print(f"Estimated beta and mu for the entire dataset: beta={beta}, mu={mu}")

    # Estimate beta and mu for the first 20 days
    beta_20, mu_20 = estimate_sir_parameters(pd.DataFrame(infected_recovered_data[:20]))
    print(f"Estimated beta and mu for the first 20 days: beta={beta_20}, mu={mu_20}")

    # Estimate beta and mu from day 20 to the end
    beta_20_end, mu_20_end = estimate_sir_parameters(pd.DataFrame(infected_recovered_data[20:]))
    print(f"Estimated beta and mu from day 20 to the end: beta={beta_20_end}, mu={mu_20_end}")