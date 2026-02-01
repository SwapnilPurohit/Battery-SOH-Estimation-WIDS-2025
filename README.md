## Week 1: Physics-Informed Neural Networks (PINNs)

The goal of this week was to build Neural Networks that can solve Ordinary Differential Equations (ODEs) by using the equation itself as the Loss Function ("Physics Loss").

### `task1_exp_decay.ipynb` (Exponential Decay)
This task implements a basic "Zero-Shot" PINN to solve the first-order differential equation for exponential decay: `dy/dt = -ky`.
* **The Approach:** Instead of training on a dataset of known answers, the network was trained to minimize the residual of the equation.
* **The Result:** The model successfully learned the function `y = e^{-kt}` purely from physics constraints, matching the analytical solution perfectly.

### `task2_shm_comparison.ipynb` (Simple Harmonic Motion)
A second-order ODE representing a Spring-Mass system (`y'' + ky = 0`). This task involved extensive experimentation to optimize PINN performance.
* **Comparison of Activation Functions:** I compared **Tanh** vs. **Sigmoid**.
    * *Result:* **Tanh** proved to be superior for this physics problem. Since the solution is an oscillating wave (going from -1 to 1), Tanh's zero-centered output works naturally. Sigmoid (0 to 1) struggled to learn the negative half of the cycles and suffered from vanishing gradients.
* **Data-assisted vs. Pure Physics:** I ran experiments comparing a standard PINN against one aided by **3 sparse data points**.
    * *Result:* The Pure PINN often got stuck in a trivial solution (a flat line at y=0) or failed to oscillate. Adding just 3 known points anchored the model, allowing it to converge rapidly to the correct Sine/Cosine wave.


## Week 2: Battery Theory & Simulation

The goal of this week was to transition from generic math to real-world battery engineering using the **PyBaMM** (Python Battery Mathematical Modelling) library.

### `task1_pybamm_1C.ipynb` (1C Discharge Animation)
This task uses the **Single Particle Model (SPM)** to simulate a 1-hour discharge at a 1C rate.
* **Visualization:** Instead of static snapshots, this task implements a **FuncAnimation** to create a real-time visualization of the discharge process.
* **Dual Tracking:** The animation simultaneously tracks:
    1.  **Terminal Voltage:** Showing the voltage drop from ~4.2V to the cutoff.
    2.  **Anode Concentration:** Tracking the normalized X-averaged negative particle concentration as it depletes over time.
* **Key Insight:** Provides a clear visual correlation between the internal electrochemical depletion of the anode and the macroscopic voltage response of the cell.

### `task2_ocv.ipynb` (OCV-SOC Curve Fitting)
This task extracts the **Open Circuit Voltage (OCV)** curve, which serves as the "fingerprint" of the battery's chemistry.
* **Method:** Instead of a continuous discharge, I ran a series of "Rest" simulations at discrete State of Charge (SOC) levels (from 0% to 100%). This allows the voltage to settle to its true thermodynamic potential without internal resistance effects.
* **Polynomial Regression:** The resulting OCV vs. SOC data points were fitted to a **5th-degree Polynomial**.
* **Application:** This polynomial `V = f(SOC)` provides a lightweight, computationally efficient way to map SOC to Voltage (and vice versa) for onboard Battery Management Systems (BMS), avoiding the need to run heavy physics simulations in real-time.

## Week 3: Data-Driven SOH Estimation (NASA Dataset)

This week shifted focus from pure physics simulation to **Data-Driven Modeling** using the **NASA Battery Aging Dataset**. The goal was to build a machine learning model capable of estimating State of Health (SOH) from raw sensor data.

### `data_processing.ipynb` (Data Pipeline & Engineering)
Before training, raw NASA files were processed into structured tensors suitable for Neural Networks.
* **Dual-Resolution Datasets:** I generated two distinct versions of the data:
    1.  **Binned Dataset (N=20):** Downsampled input for the Neural Network to efficiently learn temporal patterns.
    2.  **Physics Dataset (N=200):** High-resolution resampling used to calculate "Ground Truth" values for Charge ($Q$) and Energy ($E$), which will be used later for Physics-Informed loss functions.
* **Feature Engineering:** Extracted and aligned **Voltage**, **Current**, and **Temperature** profiles for every discharge cycle.

### `week3_transformer.ipynb` (Transformer for SOH)
Implemented a modern **Transformer Architecture** to predict SOH, replacing traditional RNNs/LSTMs.
* **Architecture:** A **Sequence-to-One** Transformer Encoder that takes the 20-step discharge profile `(V, I, T)` as input and outputs a single SOH value.
* **Why Transformers?** Unlike simple Feed-Forward networks, the Transformer uses **Self-Attention** to weigh specific parts of the voltage curve (like the "knee point") that are most indicative of aging.
* **Baseline Performance:** Established a purely data-driven baseline (training on MSE Loss) to benchmark against future Physics-Informed versions.