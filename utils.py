import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
from itertools import permutations
from hmmlearn import hmm
from matplotlib import cm

# generate colors
def get_colors(N):
    colors = cm.winter(np.linspace(0, 1, N))  # Generate colors for each hidden state
    # make each color transparent
    colors_alpha = colors.copy()
    for i in range(N): colors_alpha[i][3] = 0.5
    # convert to rgba strings
    colors = [f'rgba{tuple(color)}' for color in colors]
    colors_alpha = [f'rgba{tuple(color)}' for color in colors_alpha]
    return colors, colors_alpha

# Function to generate synthetic data
@st.cache_data
def generate_observations(N, T, p_same, max_mean, vars_same, randomize, seed):
    if randomize:
        transition_matrix = np.random.rand(N, N)
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, None]
        emission_means = np.random.uniform(-5, 5, N).reshape(-1, 1)
        emission_vars = np.random.uniform(0.1, 3, N).reshape(-1, 1)
        initial_state_distribution = np.random.dirichlet(np.ones(N))
    else:
        transition_matrix = np.ones((N, N)) * ((1 - p_same) / (N - 1))
        transition_matrix[np.diag_indices(N)] = p_same
        emission_means = np.linspace(-max_mean, max_mean, N).reshape(-1, 1)
        emission_vars = np.ones((N, 1)) * vars_same
        initial_state_distribution = np.array([1 / N] * N)
    
    hidden_states = []
    current_state = np.random.choice(N, p=initial_state_distribution)
    hidden_states.append(current_state)
    hidden_states.extend(np.random.choice(N, p=transition_matrix[hidden_states[-1]]) for _ in range(1, 2*T))
    observations = [np.random.normal(emission_means[state], np.sqrt(emission_vars[state]))[0] for state in hidden_states]

    hidden_states = np.array(hidden_states)
    observations = np.array(observations)

    data = {
        'obs_train': observations[:T],
        'obs_forecast': observations[T:],
        'hid_states_train': hidden_states[:T],
        'hid_states_forecast': hidden_states[T:],
        'pi': initial_state_distribution,
        'means': emission_means,
        'vars': emission_vars,
        'A': transition_matrix,
    }

    return data

# Visualize the time series data
def plot_time_series(observations, hidden_states, N):
    T_max_display = 1000
    colors, _ = get_colors(N)

    observations = observations[:T_max_display] if len(observations) > T_max_display else observations
    hidden_states = hidden_states[:T_max_display] if len(hidden_states) > T_max_display else hidden_states

    obs = [[observations[i] if hidden_states[i] == state else np.nan for i in range(len(observations))] for state in range(N)]

    fig = go.Figure()

    for state in range(N):
        fig.add_trace(go.Scatter(
            x=np.arange(len(observations)), 
            y=obs[state], 
            mode='markers', 
            name=f'State {state}', 
            marker=dict(
                color = colors[state], 
                size=7,
                line=dict(width=1, color='black'),
                ),
        ))
    fig.update_layout(
        title='Hidden States and Observations',
        xaxis_title='Time Step',
        yaxis_title='Observation Value',
        legend=dict(
            x=0,
            y=1,
        ),
    )
    return fig

# Function to plot histogram of synthetic data
def plot_histogram_synthetic_data(observations, hidden_states, emission_means, emission_vars, N):
    colors, colors_alpha = get_colors(N)

    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.9, 0.1],
    )
    for state in range(N):
        state_observations = observations[hidden_states == state]
        fig.add_trace(go.Histogram(
            x=state_observations, 
            name=f'State {state}', 
            marker_color= colors[state],
            opacity=0.5,
            histnorm='probability density',   #other option: 'probability' or 'density' or 'probability density' or 'percent' or 'sum' or 'sum normalized'
        ) , row=1, col=1)

        # draw the true distribution
        x = np.linspace(min(state_observations), max(state_observations), 100)
        y = (1 / np.sqrt(2 * np.pi * emission_vars[state])) * np.exp(-0.5 * (x - emission_means[state])**2 / emission_vars[state])
        fig.add_trace(go.Scatter(
            x=x, 
            y=y * len(state_observations) * np.diff(np.unique(state_observations)).mean(), 
            mode='lines', 
            name=f'True State {state} Distribution',
            line=dict(
                color=colors[state],
                ),
            showlegend=False,
            fill='tozeroy',
            fillcolor=colors_alpha[state],
        ), row=2, col=1)
    
    fig.update_layout(
        title='Observation Distributions for Each Hidden State',
        yaxis_title='Probability Density',  # y axis title for the first subplot
        # x axis title for the second subplot
        xaxis2_title='Observation Value',
        # hide y axis values for the second subplot
        yaxis2_showticklabels=False,
        barmode='overlay',
        # legend inside the plot area
        legend=dict(
            x=0,
            y=1,
        ),
    )
    return fig
        
# Function to train HMM and visualize results
# @st.cache_data
def train_hmm(observations, N, n_iter, covariance_type):
    # apply seed for reproducibility
    # np.random.seed(seed)
    model = hmm.GaussianHMM(n_components=N, covariance_type=covariance_type, n_iter=n_iter)
    model.fit(observations.reshape(-1, 1))
    return model

# Function to visualize HMM results
def visualize_hmm(data, model, N):
    observations = data['obs_train']
    inferred_hidden_states = model.predict(observations.reshape(-1, 1))
    means_hmmlearn = model.means_
    covars_hmmlearn = model.covars_
    transition_matrix = model.transmat_

    accuracy,states = max_similarity(data['hid_states_train'], inferred_hidden_states)
    fig_accuracy = plot_accuracy(accuracy)

    # Visualize true vs inferred distributions
    x = np.linspace(-10, 10, 1000)
    fig_main = make_subplots(
        rows=1, cols=2,
        column_widths=[0.8, 0.2], 
        subplot_titles=('State Distributions', 'Confusion Matrix'),
    )
    colors = cm.winter(np.linspace(0, 1, N))
    colors = [f'rgba{tuple(color)}' for color in colors]

    for state in range(N):
        y_true = norm.pdf(x, data['means'][state], np.sqrt(data['vars'][state]))
        y_inferred = norm.pdf(x, means_hmmlearn[state], np.sqrt(covars_hmmlearn[state]))
        fig_main.add_trace(go.Scatter(
            x=x,
            y=y_true,
            mode='lines',
            name=f'True States',
            showlegend = state == 0,
            line=dict(color=colors[0]),
        ), row=1, col=1)
        fig_main.add_trace(go.Scatter(
            x=x,
            y=y_inferred[0],
            mode='lines',
            name=f'Inferred States',
            showlegend = state == 0,
            line=dict(color=colors[-1], dash='dash'),
        ), row=1, col=1)

    # change the label of hidden states in inferred_hidden_states from 0, 1, 2 to the corresponding state in the true hidden states
    inferred_hidden_states = np.array([states[i] for i in inferred_hidden_states])
    conf = confusion_matrix(data['hid_states_train'], inferred_hidden_states)
    fig_main.add_trace(go.Heatmap(
        z=conf, 
        colorscale='Viridis',
        showscale=False,
        text=conf,
        texttemplate='%{text:.0f}',
        hoverinfo='skip',
    ), row=1, col=2)
    fig_main.update_layout(
        title='Comparison of True and Inferred State Distributions',
        xaxis_title='Observation Value',
        yaxis_title='Probability Density',
        legend=dict(
            x=0,
            y=1,
        ),
        # title2='Confusion Matrix of Inferred States',
        xaxis2_title='Inferred States',
        yaxis2_title='True States',
        xaxis2=dict(scaleanchor="y2", scaleratio=1),
    )

    fig_trans = make_subplots(rows=1, cols=2, subplot_titles=('True Transition Matrix', 'Inferred Transition Matrix'))
    fig_trans.add_trace(go.Heatmap(
        z=data['A'],
        colorscale='Viridis',
        showscale=False,
        text=data['A'],
        texttemplate='%{text:.1%}',
        hoverinfo='skip',
    ), row=1, col=1)
    fig_trans.add_trace(go.Heatmap(
        z=transition_matrix,
        colorscale='Viridis',
        showscale=False,
        text=transition_matrix,
        texttemplate='%{text:.1%}',
        hoverinfo='skip',
    ), row=1, col=2)
    fig_trans.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        xaxis2=dict(scaleanchor="y2", scaleratio=1),
        # x and y axis titles should be from/to states for both subplots
        xaxis_title='From State',
        yaxis_title='To State',
        xaxis2_title='From State',
        yaxis2_title='To State',
        
    )

    figs = [fig_main, fig_accuracy, fig_trans]
    return figs

# Function to calculate max similarity between two sequences
def max_similarity(a, b):
    unique_values = np.unique(a)
    N = len(unique_values)
    value_to_index = {v: i for i, v in enumerate(unique_values)}
    a_indices = np.array([value_to_index[v] for v in a])
    b_indices = np.array([value_to_index[v] for v in b])
    all_perms = permutations(range(N))
    max_accuracy = 0
    states = []
    for perm in all_perms:
        permuted_b_indices = np.array([perm[i] for i in b_indices])
        accuracy = np.mean(a_indices == permuted_b_indices)
        max_accuracy = max(max_accuracy, accuracy)
        states = perm if accuracy == max_accuracy else states
    return max_accuracy, states

# Function to plot accuracy of inferred states
def plot_accuracy(accuracy):
    # plot a horizontal bar chart of accuracy score that also shows 100% line
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[accuracy],
        y=['Accuracy'],
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.6)'),
        text=[f'{accuracy:.1%}'],
        # text color
        textfont=dict(color='white'),
        textposition = 'inside',
        name='Accuracy',
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=[1],
        y=['Accuracy'],
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.3)'),
        showlegend=False,
    ))
    fig.update_layout(
        title='Accuracy of Inferred States',
        xaxis_title='Accuracy',
        yaxis_title='',
        yaxis=dict(visible=False),
        # overlay bars on top of each other
        barmode='overlay',
        # width and height of the figure
        # width=600,
        height=200,
    )
    return fig

# plot forecasts vs true emissions
def plot_actual_vs_predicted_linear(emissions_true, emissions_forecast):
    x = emissions_forecast.flatten()
    y = emissions_true.flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        showlegend=False,
        marker=dict(
            # color= colors,
            size=3,
        ),
    ))
    fig.update_xaxes(
        # same scale for x and y axes
        scaleanchor="y",
    )
    fig.add_trace(go.Scatter(
        x=[np.min(emissions_true), np.max(emissions_true)],
        y=[np.min(emissions_true), np.max(emissions_true)],
        mode="lines",
        line=dict(
            dash="dash",
            color="gray",
            width=1,
        ),
        showlegend=False
    ))
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )
    
    return fig

# Function to forecast future emissions
def forecast_and_evaluate(observations, model):
    forecasts = []  # To store forecasted emissions
    # errors = []     # To store absolute errors between forecasted and actual emissions
    hidden_states_list = []  # To store inferred hidden states
    # Iterate through the observation array
    for i in range(len(observations) - 1):
        # Current observation (today's emission)
        current_observation = observations[i]
        
        # Use the HMM to infer the most likely hidden state for today
        _, hidden_states = model.decode(np.array([[current_observation]]))
        current_hidden_state = hidden_states[0]  # Most likely hidden state
        
        # Predict the next hidden state using transition probabilities
        next_hidden_state_probs = model.transmat_[current_hidden_state]
        next_hidden_state = np.argmax(next_hidden_state_probs)  # Most likely next hidden state
        hidden_states_list.append(next_hidden_state)

        # Predict the next day's emission using the emission probabilities of the next hidden state
        next_emission_mean = model.means_[next_hidden_state][0]  # Mean of Gaussian distribution
        forecasts.append(next_emission_mean)
        
        # Compare the forecasted emission with the actual observation for the next day
        # actual_next_observation = observations[i + 1]
        # error = abs(next_emission_mean - actual_next_observation)
        # errors.append(error)
    st.write(hidden_states_list)
    return np.array(forecasts)        #, errors
