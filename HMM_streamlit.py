import streamlit as st
import numpy as np
import utils
st.set_page_config(layout="wide")


# initialize st.session_state
if 'data' not in st.session_state:
    st.session_state.data = False
if 'training' not in st.session_state:
    st.session_state.training = False
    
# Sidebar
# Section 1: Generate Synthetic Data
st.sidebar.title("Synthetic Data Generation Parameters")
N = st.sidebar.number_input("Number of Hidden States, $N$", min_value=2, value=3)
T = st.sidebar.number_input("Number of Time Steps, $T$", min_value=100, value=1000)
randomize = st.sidebar.checkbox("Randomize Data Generation", value=False)
if randomize:
    seed = st.sidebar.number_input("Random Seed", min_value=0, value=0)
    np.random.seed(seed)
else:
    p_same = st.sidebar.slider("Probability of Staying in the Same State, $A_{ii}$", 0.1, 0.95, 0.5)
    max_mean = st.sidebar.slider("Maximum and Minimum Mean Value for Emission Distribution, $\mu_\\text{max}$", 0.1, 7.0, 2.0)
    vars_same = st.sidebar.slider("Variance for Emission Distributions, $\sigma^2$", 0.1, 2.0, 0.5)


# Main Streamlit app
st.title("Hidden Markov Model (HMM) Simulator")
st.header("1. What is a Hidden Markov Model?")
st.write("""
A **Hidden Markov Model (HMM)** is a statistical model that represents systems with hidden (unobservable) states that influence observable outcomes. 
HMMs are widely used in fields such as speech recognition, bioinformatics, finance, and natural language processing.
""")
st.write("""
At its core, an HMM assumes that the system being modeled transitions between a set of **hidden states** over time, and each hidden state generates 
an **observation** or **emission** according to some probability distribution. The key idea is that while we can observe the generated data, the underlying hidden states 
are not directly visible.
""")

st.subheader("Mathematical Representation")
st.write("""
An HMM is defined by the following components:
1. **Number of hidden states (`N`)**: Denoted as $ S_1, S_2, \dots, S_N $.
2. **Transition probabilities**: The probability of transitioning from one hidden state to another is represented by a matrix $ A $, where:
   $$
   A_{ij} = P(S_t = j \mid S_{t-1} = i)
   $$
   This means $ A_{ij} $ is the probability of transitioning from state $ i $ at time $ t-1 $ to state $ j $ at time $ t $.
3. **Emission probabilities**: Each hidden state $ S_i $ generates an observation $ O_t $ according to a probability distribution. For example, if the observations 
   are continuous, the emission probabilities might follow a Gaussian distribution:
   $$
   P(O_t \mid S_t = i) = \mathcal{N}(O_t \mid \mu_i, \sigma_i^2)
   $$
   Here, $ \mu_i $ and $ \sigma_i^2 $ are the mean and variance of the Gaussian distribution for state $ i $.
4. **Initial state distribution**: The probability of starting in each hidden state is given by $ \pi $, where:
   $$
   \pi_i = P(S_1 = i)
   $$

The goal of an HMM is to infer the hidden states $ S_t $ given the observed data $ O_1, O_2, \dots, O_T $.

It should be noted that not every dynamic system in the universe can be accurately modeled or explained using Hidden Markov Models (HMMs). 
HMMs assume that the underlying system can be described by a finite set of hidden states, with transitions between these states following a Markov process (i.e., the next state depends only on the current state and not on the sequence of events that preceded it). 
Additionally, HMMs assume that observations are generated independently from the current hidden state. 
While this framework is powerful for modeling systems with discrete states and observable outputs (e.g., speech recognition, DNA sequencing), it falls short for systems with continuous, non-Markovian dynamics, or those governed by complex interactions and memory effects. 
For example, chaotic systems, quantum mechanical phenomena, or systems with long-range dependencies often defy the simplifying assumptions of HMMs. 
Thus, while HMMs are a valuable tool for specific applications, they are not universally applicable to all dynamic systems.
""")


st.header("2. Generating Synthetic Data Resembling HMM Systems")
st.write("""
To understand how HMMs work, it’s helpful to simulate synthetic data that mimics real-world systems governed by hidden states. Here’s how we can generate such data:
""")
st.write("""
1. **Define the number of hidden states (`N`)**:

    Each hidden state represents a distinct mode of behavior or condition in the system.
    This can be set using the slider in the sidebar.
""")

st.write("""
2. **Define the number of time steps**:

    The length of the time series data to be generated. 
    This can be set using the slider in the sidebar.
""")

st.write("""
3. **Create a transition matrix (`A`)**:

    The transition matrix specifies the probability of moving from one hidden state to another. For example, if $ N = 3 $, the transition matrix might look like this:
    $$
    A = \\begin{bmatrix}
    0.7 & 0.2 & 0.1 \\\\
    0.3 & 0.5 & 0.2 \\\\
    0.1 & 0.4 & 0.5
    \\end{bmatrix}
    $$
         
    This means there’s a 70% chance of staying in state 1, a 20% chance of transitioning to state 2, and so on.
    The sum of each row in the transition matrix should be 1, ensuring that the system moves to a new state at each time step.
    Here, we generate synthetic data assuming a simple transition matrix such that:
    - There is a fixed probability of staying in the same state for each hidden state.
    - The remaining probability is distributed uniformly among the other states.

    So to create the transition matrix, we only need to specify the probability of staying in the same state.
    This can be changed using the slider in the sidebar.
""")

st.write("""
4. **Specify emission distributions**:

    Each hidden state generates observations based on a probability distribution. For simplicity, we often use Gaussian distributions:
    $$
    P(O_t \mid S_t = i) = \mathcal{N}(\mu_i, \sigma_i^2)
    $$
    For example, state 1 might have $ \mu_1 = -1 $ and $ \sigma_1^2 = 0.5 $, while state 2 has $ \mu_2 = 1 $ and $ \sigma_2^2 = 0.5 $.
    Here, we make a couple of assumptions to simplify the generation of synthetic data:
    - The mean values $ \mu_i $ are evenly spaced between $ -\mu_{\\text{max}} $ and $ \mu_{\\text{max}} $. 
    The value of $ \mu_{\\text{max}} $ can be adjusted using the slider in the sidebar.
    - The variance $ \sigma_i^2 $ is the same for all hidden states and can be adjusted using the slider in the sidebar.
""")
st.write("""
5. **Simulate the process**:

    To generate synthetic data, we follow these steps:
    - Start with an initial hidden state sampled from the initial distribution $ \pi $.
    - At each time step:
        - Transition to a new hidden state based on $ A $.
        - Generate an observation from the emission distribution corresponding to the current hidden state.

    This process produces a sequence of observations $ O_1, O_2, \dots, O_T $ and a corresponding sequence of hidden states $ S_1, S_2, \dots, S_T $. 
    While the observations are visible, the hidden states remain unknown to the observer.
""")
if not randomize: seed=0    # dummy value
if randomize: p_same, max_mean, vars_same = 0,0,0 # dummy values    
data = utils.generate_observations(N, T, p_same, max_mean, vars_same, randomize, seed)
st.session_state.data = True

if st.session_state.data:
    st.success("Synthetic data generated successfully!")
fig_time = utils.plot_time_series(
    data['obs_train'],
    data['hid_states_train'], 
    data['prices'],
    N,
)
fig_hist_true = utils.plot_histogram_synthetic_data(
    data['obs_train'], 
    data['hid_states_train'], 
    data['means'], 
    data['vars'], 
    N)
st.plotly_chart(fig_time)
st.plotly_chart(fig_hist_true)

st.header("3. Using HMM to Infer Hidden States")
st.write("""
Once we have synthetic data, the next step is to train an HMM to infer the hidden states. This involves two main tasks:
""")
st.write("""
1. **Learning the parameters of the HMM**:
    - Given only the observed data $ O_1, O_2, \dots, O_T $, we want to estimate:
        - The transition matrix $ A $.
        - The emission distributions (e.g., means $ \mu_i $ and variances $ \sigma_i^2 $).
        - The initial state distribution $ \pi $.
    - This is typically done using the **Baum-Welch algorithm**, which is a variant of the Expectation-Maximization (EM) algorithm. It iteratively updates the model 
    parameters to maximize the likelihood of the observed data.
""")
st.write("""
2. **Decoding the hidden states**:
   - After training the HMM, we can infer the most likely sequence of hidden states $ S_1, S_2, \dots, S_T $ that generated the observations. This is done using 
     the **Viterbi algorithm**, which finds the optimal sequence of hidden states by maximizing the joint probability of the observations and the hidden states.
""")

st.subheader("Steps in Practice")
st.write("""
- **Step 1: Train the HMM**:
  - Provide the observed data to the HMM and let it learn the parameters (transition matrix, emission distributions, etc.).
  - The trained model will approximate the true parameters used to generate the synthetic data.
""")
st.write("""
- **Step 2: Evaluate the inferred states**:
  - Compare the inferred hidden states with the true hidden states used during data generation.
  - Since the labels of the hidden states may differ between the true and inferred sequences (e.g., state 1 in the true sequence might correspond to state 3 in the 
    inferred sequence), we account for label ambiguity by calculating the maximum similarity between the two sequences.
""")
st.write("""
##### **Why Use Synthetic Data?**
Synthetic data allows us to test and validate HMM algorithms in a controlled environment. By knowing the true hidden states, we can evaluate the accuracy of the 
inferred states and gain insights into the strengths and limitations of HMMs.
""")


# Section 2: Train HMM
with st.form(key='train_hmm'):
    st.subheader("HMM Training Parameters")
    n_iter = st.number_input("Number of Iterations", min_value=10, value=500)
    covariance_type = st.selectbox("Covariance Type", ["full", "spherical", "diag", "tied"], index=0)
    # seed = st.number_input("Random Seed", min_value=0, value=0)
    # st.subheader("Forecasting Parameters")
    # num_samples = st.number_input("Number of Trajectories to Generate", min_value=1, value=5)
    submit_button = st.form_submit_button(label='Train HMM')

if submit_button:
    model = utils.train_hmm(
        data['obs_train'], 
        N,
        n_iter,
        covariance_type)
    st.success("HMM trained successfully!")
    st.session_state.training = True
    
    st.subheader("HMM Inference Results")
    figs = utils.visualize_hmm(data, model, N)
    fig_main, fig_accuracy, fig_trans = figs
    st.write("""
    The plots below show the comparison between the true and inferred state distributions, the accuracy of the inferred states, and the confusion matrix of the inferred states.
    Confusion matrix is a table that shows the number of correct and incorrect predictions made by the model.
             
    Accuracy of the inferred states is calculated as the ratio of the number of correctly inferred states to the total number of states.
    """)
    st.plotly_chart(fig_main)

    # Evaluate accuracy
    st.plotly_chart(fig_accuracy)

    # Visualize transition matrices
    st.plotly_chart(fig_trans)

else:
    st.warning("Please train the HMM first to enable forecasting.")