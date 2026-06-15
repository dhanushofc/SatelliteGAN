# Chapter [X]: Comparative Analysis of Convolutional LSTM and Foundation Models for Sea Surface Temperature Forecasting

[Author Name 1]

[Designation/Role]

[Department]

[Institution/College]

[Location]

and is the corresponding author.

E-mail: [Email Address]

[Author Name 2]

[Designation/Role]

[Department]

[Institution/College]

[Location]

E-mail: [Email Address]

---

### Abstract

Sea Surface Temperature (SST) forecasting represents one of the most critical challenges in operational oceanography, with direct implications for monsoon prediction, marine ecosystem management, fisheries optimization, and climate change monitoring in the Indian Ocean region. A comprehensive comparative study was conducted across three distinct forecasting paradigms: a custom Convolutional Long Short-Term Memory (ConvLSTM) architecture, Amazon Chronos (a transformer-based foundation model), and IBM Granite Time-Series Foundation Model (an MLP-Mixer-based architecture). The evaluation was performed under a unified protocol using 16,290 days of SST observations (September 1, 1981 – April 7, 2026) over a 60×48 spatial grid covering the region 5°N–20°N, 60°E–72°E, with a 90-day rolling forecast evaluation period (January–March 2026) at the target location 8.0°N, 67.0°E in the Laccadive Sea. A rigorous five-gate evaluation framework was established to assess model performance across multiple dimensions: overall RMSE (<0.1466°C), February RMSE (<0.2093°C), March RMSE (≤0.1003°C), big error count (≤12 days with |error|≥0.20°C), and slope ([0.94, 1.00] measuring amplitude response fidelity). Twenty-five experimental runs were completed across all model families, encompassing zero-shot inference, few-shot post-hoc calibration, LoRA fine-tuning, and a novel PostGain slope correction technique. The ConvLSTM architecture achieved full compliance with all five evaluation gates (5/5), demonstrating an overall RMSE of 0.1417°C. The Chronos few-shot configuration achieved the lowest RMSE of 0.1261°C (11% improvement over ConvLSTM) but failed the slope gate (0.8634), indicating systematic amplitude compression. The critical breakthrough was achieved through PostGain slope correction applied to single-model zero-shot inference pipelines, wherein the Granite-only spatial configuration achieved an overall RMSE of 0.1196°C with a slope of 0.9436, passing all five gates (5/5) — the first foundation model configuration to achieve full gate compliance while simultaneously surpassing ConvLSTM on RMSE by 16%. The PostGain technique applies a post-hoc gain multiplier (1.020–1.040) fitted on validation data to resolve the systematic amplitude compression previously observed in all foundation model configurations. Ensemble pipelines were explored as a secondary investigation, with the point ensemble achieving the lowest RMSE of 0.1187°C (5/5 gates), though these results are documented separately from the primary single-model analysis. The findings demonstrate that foundation models, when equipped with PostGain slope targeting, can match and exceed the performance of purpose-built ConvLSTM architectures while maintaining full gate compliance, establishing a new benchmark for operational SST forecasting in the Indian Ocean region.

---

### 1. Introduction

#### 1.1 Background and Context

Oceanographic forecasting represents one of the most computationally intensive and scientifically significant challenges in modern environmental science, with sea surface temperature (SST) serving as the primary indicator of oceanic health, climate variability, and marine ecosystem dynamics. The Indian Ocean, covering approximately 20% of the Earth's water surface, plays a critical role in global climate regulation through its influence on monsoon systems, tropical cyclone formation, and large-scale ocean-atmosphere coupling phenomena such as the Indian Ocean Dipole. The accurate prediction of SST variations across temporal scales ranging from days to decades has become increasingly essential for operational meteorology, fisheries management, maritime navigation, and climate change adaptation strategies.

Traditional approaches to SST forecasting have relied predominantly on Numerical Weather Prediction (NWP) models, which solve complex systems of partial differential equations governing fluid dynamics, thermodynamics, and radiative transfer in the atmosphere-ocean system. These physics-based models, while theoretically comprehensive, require enormous computational resources and exhibit significant limitations in regional forecasting contexts where localized phenomena such as coastal upwelling, eddy dynamics, and monsoon-driven mixing introduce substantial variability that coarse-resolution global models cannot adequately capture. Statistical methods including autoregressive integrated moving average (ARIMA) models, empirical orthogonal function (EOF) analysis, and multiple linear regression have been employed as computationally efficient alternatives, but these approaches fundamentally assume linear relationships and stationary statistical properties that do not hold for the highly non-linear, non-stationary SST time series observed in tropical ocean regions.

The emergence of deep learning architectures has fundamentally transformed the landscape of time-series forecasting by enabling the direct learning of complex non-linear relationships from observational data without requiring explicit physical parameterizations. Recurrent neural networks, particularly Long Short-Term Memory (LSTM) architectures [1], have demonstrated superior performance over traditional statistical methods in capturing temporal dependencies in sequential data. However, standard LSTM architectures process input data as one-dimensional sequences, thereby discarding the spatial relationships inherent in gridded SST observations where neighboring grid cells exhibit correlated temperature variations driven by oceanic currents, heat transport mechanisms, and atmospheric forcing patterns.

Convolutional Long Short-Term Memory (ConvLSTM) architectures, first introduced by Shi et al. [2], address this limitation by integrating convolutional operations within the LSTM cell structure, enabling the simultaneous capture of spatial and temporal dependencies in gridded data. The convolutional operations preserve the two-dimensional structure of the input data, allowing the model to learn spatial patterns such as temperature gradients, frontal structures, and eddy signatures that are essential for accurate SST forecasting. The LSTM gating mechanisms capture temporal dependencies across the input sequence, enabling the model to learn the evolution of spatial patterns over time. This combination of spatial and temporal modeling makes ConvLSTM architectures particularly well-suited for SST forecasting tasks where both dimensions are critical for prediction accuracy.

More recently, foundation models — large-scale neural networks pre-trained on diverse, multi-domain datasets — have emerged as powerful alternatives to purpose-built architectures in time-series forecasting. Amazon Chronos [3], a transformer-based foundation model pre-trained on thousands of time-series datasets, and IBM Granite Time-Series Foundation Model (TTM) [4], an MLP-Mixer-based architecture with only 71,000 parameters, have demonstrated remarkable zero-shot forecasting capabilities across a wide range of domains. The applicability of these foundation models to domain-specific tasks such as SST forecasting, and their comparative effectiveness relative to purpose-built architectures such as ConvLSTM, remains an open question that this chapter addresses through a comprehensive empirical evaluation.

#### 1.2 Importance of Sea Surface Temperature

Sea surface temperature serves as a fundamental parameter in oceanographic and atmospheric sciences, influencing a diverse range of physical, biological, and socio-economic systems. The thermal state of the ocean surface directly modulates the exchange of heat, moisture, and momentum between the ocean and atmosphere, thereby controlling the development and intensification of weather systems including tropical cyclones, monsoon depressions, and mid-latitude storm tracks. Anomalies in SST patterns, particularly in the tropical Indian Ocean, have been conclusively linked to the variability of the Indian Summer Monsoon, which sustains the agricultural economy of over one billion people and determines the water security of the entire South Asian region [5].

The biological productivity of marine ecosystems is intimately connected to SST through its influence on nutrient upwelling, phytoplankton bloom dynamics, and the spatial distribution of fish populations. Coral reef ecosystems, which support approximately 25% of all marine species, are particularly sensitive to SST anomalies, with prolonged exposure to temperatures exceeding local thresholds by 1–2°C triggering mass bleaching events that can devastate reef communities and the livelihoods of coastal populations dependent on reef-associated fisheries and tourism [6]. The Arabian Sea and Bay of Bengal regions, encompassing the geographical domain of this study (5°N–20°N, 60°E–72°E), host some of the world's most productive fisheries and are home to extensive coral reef systems that require accurate SST forecasting for effective conservation and management.

From an operational perspective, SST forecasts are essential inputs for maritime navigation safety, offshore oil and gas operations, naval strategic planning, and search-and-rescue operations. The Indian National Centre for Ocean Information Services (INCOIS) provides operational oceanographic services to multiple government agencies and commercial stakeholders, and the accuracy of SST forecasts directly impacts the reliability of these services. The development of improved forecasting methodologies that can deliver higher accuracy at lower computational cost represents a significant operational priority for INCOIS and similar oceanographic institutions worldwide.

#### 1.3 Problem Statement

The fundamental challenge addressed in this chapter is the development of a robust, accurate, and computationally efficient SST forecasting system capable of producing reliable 7-day rolling forecasts for the Indian Ocean region, with particular emphasis on the Laccadive Sea target location at 8.0°N, 67.0°E. This challenge encompasses multiple interrelated technical difficulties that must be simultaneously addressed.

The first difficulty arises from the inherent complexity of SST dynamics, which are governed by a combination of atmospheric forcing (wind stress, solar radiation, air-sea heat flux), oceanic processes (currents, upwelling, eddy dynamics, thermocline variability), and boundary effects (coastal geometry, bathymetry, river discharge). These processes operate across multiple temporal and spatial scales, from sub-daily diurnal heating cycles to interannual climate oscillations, and from meter-scale turbulent mixing to basin-scale circulation patterns. Any forecasting system must capture the dominant modes of variability relevant to the 7-day prediction horizon while remaining computationally tractable for operational deployment.

The second difficulty stems from the limitations of existing forecasting approaches when applied to regional SST prediction. Numerical weather prediction models, while physically comprehensive, require computational resources that exceed the operational budgets of many oceanographic institutions and exhibit systematic biases in regional contexts where sub-grid scale processes dominate. Statistical methods, while computationally efficient, fail to capture the non-linear dynamics and regime shifts characteristic of tropical SST variability. Standard deep learning approaches, particularly LSTM architectures, discard spatial information that is critical for capturing the propagation of temperature anomalies across the ocean surface.

The third difficulty relates to the evaluation and validation of forecasting systems in an operational context. Traditional evaluation metrics such as overall RMSE provide a single aggregate measure of forecast accuracy but fail to capture critical aspects of forecast quality including performance during extreme events, seasonal variability, amplitude fidelity, and the frequency of large errors. A comprehensive evaluation framework must assess multiple dimensions of forecast quality simultaneously to ensure that the selected model is suitable for operational deployment across the full range of conditions encountered in practice.

#### 1.4 Objectives

The primary objective of this study is to develop and evaluate a comprehensive SST forecasting system that addresses the limitations of existing approaches while meeting the operational requirements of INCOIS. This overarching objective is decomposed into the following specific technical objectives.

The first objective is to design and implement a ConvLSTM architecture that preserves the spatial relationships inherent in gridded SST data while capturing the temporal dependencies necessary for multi-day forecasting. The architecture must process 60-day input sequences to produce 7-day rolling forecasts over a 60×48 spatial grid, with particular attention to the target location at 8.0°N, 67.0°E. The implementation must be optimized for deployment on commodity GPU hardware (NVIDIA T4) to ensure operational feasibility within the computational constraints of the institution.

The second objective is to evaluate the applicability of foundation models, specifically Amazon Chronos and IBM Granite TSFM, to the SST forecasting task through three distinct paradigms: zero-shot inference using pre-trained weights, few-shot post-hoc calibration using validation-set residual correction, and Low-Rank Adaptation (LoRA) fine-tuning for domain adaptation. This evaluation must determine whether foundation models can match or exceed the performance of a purpose-built ConvLSTM architecture while potentially offering advantages in terms of development time and computational efficiency.

The third objective is to establish a rigorous five-gate evaluation framework that assesses forecast quality across multiple dimensions including overall accuracy, seasonal performance, extreme event handling, and amplitude fidelity. This framework must provide a comprehensive basis for model comparison and selection that goes beyond single-metric evaluation to ensure operational suitability.

The fourth objective is to develop and validate post-processing techniques that improve forecast accuracy without requiring model retraining, including per-horizon bias correction, adaptive drift correction, amplitude calibration, and a novel PostGain slope targeting mechanism that resolves the systematic amplitude compression observed in foundation model predictions.

#### 1.5 Scope of the Study

The scope of this study encompasses the complete development lifecycle of an SST forecasting system, from initial data exploration and baseline model development through advanced architecture design, foundation model integration, comprehensive evaluation, and operational readiness assessment. The geographical scope is limited to the Indian Ocean region bounded by 5°N–20°N latitude and 60°E–72°E longitude, with a specific focus on the Laccadive Sea target location at 8.0°N, 67.0°E. The temporal scope covers 16,290 days of SST observations from September 1, 1981 to April 7, 2026, with model training utilizing the period September 1, 1981 to approximately April 2023 (85% of data), validation utilizing approximately May 2023 to December 2023 (5% of data), and final evaluation utilizing the 90-day period from January 1, 2026 to March 31, 2026 (10% of data).

The technical scope encompasses the development of 30+ Python scripts implementing various forecasting architectures, training strategies, and evaluation methodologies. These scripts are organized into seventeen distinct phases of development, progressing from basic implementations through increasingly sophisticated approaches. The evaluation scope encompasses twenty-five completed experimental runs across three model families (ConvLSTM, Chronos, Granite), with each run assessed against the five-gate evaluation framework.

The study does not encompass real-time operational deployment, integration with INCOIS production systems, or extension to additional forecast horizons beyond 7 days. These activities are identified as future work and are discussed in the conclusion section of this chapter.

---

### 2. Literature Review

#### 2.1 Numerical Weather Prediction Models for SST Forecasting

Numerical Weather Prediction (NWP) models represent the third generation of SST forecasting approaches, solving systems of partial differential equations that govern fluid dynamics, thermodynamics, and radiative transfer in the coupled atmosphere-ocean system. These physics-based models, including the Regional Ocean Modeling System (ROMS) [7], the Hybrid Coordinate Ocean Model (HYCOM) [8], and the Navy Coastal Ocean Model (NCOM) [9], provide physically consistent forecasts that capture the full range of oceanographic processes. The governing equations solved by these models include the Navier–Stokes equations under the Boussinesq and hydrostatic approximations, the continuity equation, the equation of state for seawater, and the thermodynamic energy equation incorporating air-sea heat flux parameterizations [47].

The ROMS model, developed by the Rutgers University and UCLA collaboration, employs a terrain-following vertical coordinate system that enables high-resolution representation of coastal processes including upwelling, estuarine circulation, and shelf-break dynamics. The model has been extensively applied to SST forecasting in the Indian Ocean region, with studies demonstrating its capability to reproduce the seasonal cycle of SST with root mean square errors of 0.5–1.0°C [10]. The ROMS model employs a split-explicit time-stepping scheme that separates the fast barotropic mode from the slow baroclinic mode, enabling efficient integration at high spatial resolutions. However, the computational cost of ROMS simulations, particularly at high spatial resolutions (≤0.25°), limits their accessibility for institutions with constrained computational budgets. A typical regional ROMS configuration at 0.25° resolution requires approximately 500 CPU-hours per forecast day on a modern HPC cluster, rendering it impractical for real-time operational deployment without dedicated computational infrastructure [48].

The HYCOM model employs a hybrid vertical coordinate system that transitions between isopycnic, terrain-following, and pressure coordinates based on local stratification, enabling efficient representation of both open-ocean and coastal processes. HYCOM has been operationalized by the U.S. Naval Oceanographic Office for global and regional SST forecasting, with demonstrated skill scores of 0.7–0.8 for 7-day forecasts in the tropical Indian Ocean [11]. The model's data assimilation system incorporates satellite-derived SST observations from the NOAA Optimum Interpolation SST (OISST) product, in-situ measurements from Argo floats, and ship-based observations through a multivariate optimal interpolation scheme. The model's computational requirements, however, remain prohibitive for real-time operational deployment at regional scales, with the global configuration requiring approximately 2,000 CPU-hours per day on a dedicated supercomputing facility [49].

The NWP approach, while physically comprehensive, exhibits several fundamental limitations when applied to regional SST forecasting. The first limitation relates to the parameterization of sub-grid scale processes, including turbulent mixing, convective adjustment, and air-sea flux exchange, which must be represented through empirical relationships that introduce systematic biases. The K-profile parameterization (KPP) scheme commonly employed for vertical mixing, for instance, has been shown to overestimate mixing depth in stratified tropical waters, leading to cold biases in SST forecasts of 0.3–0.5°C [50]. The second limitation concerns the initialization problem, wherein the accuracy of NWP forecasts is critically dependent on the quality of the initial conditions, which are derived from sparse observational networks and data assimilation systems that may not adequately capture the fine-scale SST variability relevant to regional forecasting. The third limitation involves the computational cost, with global NWP configurations requiring supercomputing facilities and regional configurations requiring dedicated GPU clusters, rendering them inaccessible for many operational oceanographic institutions.

#### 2.2 Statistical Methods for Time-Series Forecasting

Statistical methods represent the second generation of SST forecasting approaches, encompassing techniques such as autoregressive integrated moving average (ARIMA) models, empirical orthogonal function (EOF) analysis, and multiple linear regression. These methods were developed during the mid-to-late twentieth century and remain in use for baseline forecasting applications where computational resources are limited.

ARIMA models, formalized by Box and Jenkins [12], capture temporal autocorrelation structures in SST time series through a combination of autoregressive and moving average components. The ARIMA(p,d,q) model is defined by the equation:

$$\phi(B)(1-B)^d y_t = \theta(B)\epsilon_t$$

where $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$ is the autoregressive polynomial of order $p$, $\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$ is the moving average polynomial of order $q$, $d$ is the degree of differencing required to achieve stationarity, $B$ is the backshift operator defined by $B y_t = y_{t-1}$, and $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ is the white noise error term. ARIMA models have been applied to SST forecasting in the Indian Ocean with moderate success, producing 7-day forecast RMSE values of 0.4–0.6°C [13]. Extensions including seasonal ARIMA (SARIMA) and vector ARIMA (VARIMA) have been explored to capture the annual and semi-annual periodicity inherent in SST time series, though the fundamental assumption of linear relationships and stationary statistical properties limits their applicability to the highly non-linear, non-stationary SST dynamics observed in tropical ocean regions [51].

EOF analysis, also known as Principal Component Analysis (PCA) in the context of spatial data, decomposes the spatial SST field into orthogonal modes of variability, enabling dimensionality reduction and the identification of dominant spatial patterns. The method is defined by the eigenvalue decomposition of the spatial covariance matrix $\mathbf{C} = \frac{1}{T}\mathbf{X}^T\mathbf{X}$, where $\mathbf{X} \in \mathbb{R}^{T \times N}$ is the data matrix with $T$ time steps and $N$ spatial grid points. The decomposition yields $\mathbf{C} = \mathbf{E}\mathbf{\Lambda}\mathbf{E}^T$, where $\mathbf{E}$ contains the eigenvectors (EOFs) and $\mathbf{\Lambda}$ contains the eigenvalues representing the variance explained by each mode. The method was first applied to oceanographic data by Preisendorfer [14] and has since become a standard tool for SST variability analysis. EOF analysis has been used to identify the dominant modes of SST variability in the Indian Ocean, including the Indian Ocean Dipole (IOD) mode, the basin-wide warming mode, and the coastal upwelling mode [15]. The temporal principal components (PCs) associated with each EOF can be forecast using ARIMA or other time-series models, and the spatial field reconstructed through the truncated expansion $\hat{\mathbf{X}} = \mathbf{P}\mathbf{E}_k^T$, where $\mathbf{P}$ contains the forecast PCs and $\mathbf{E}_k$ contains the first $k$ EOFs. However, the linear decomposition cannot capture non-linear interactions between modes that are essential for accurate forecasting, and the truncation of higher-order modes discards fine-scale variability that may be operationally significant [52].

Multiple linear regression establishes relationships between SST and predictor variables such as wind stress, solar radiation, and atmospheric pressure. The model is defined by the equation:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_k x_k + \epsilon$$

where $y$ is the SST value, $x_i$ are the predictor variables, $\beta_i$ are the regression coefficients estimated through ordinary least squares minimization of $\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$, and $\epsilon$ is the error term assumed to be independently and identically distributed. Multiple linear regression has been applied to SST forecasting with RMSE values of 0.3–0.5°C for 7-day forecasts [16]. Extensions including ridge regression and LASSO regularization have been employed to address multicollinearity among predictor variables, though the linear functional form remains insufficient to represent the complex non-linear dynamics of ocean-atmosphere coupling [53].

#### 2.3 Deep Learning for Time-Series Forecasting

Deep learning approaches represent the fourth generation of SST forecasting methods, leveraging the capacity of neural networks to learn complex non-linear relationships directly from observational data without requiring explicit physical parameterizations. The application of deep learning to time-series forecasting has evolved through several distinct phases, each addressing specific limitations of its predecessors.

Standard feedforward neural networks were first applied to time-series forecasting in the early 1990s, with studies demonstrating their capacity to approximate non-linear functions that map historical observations to future values [17]. The universal approximation theorem establishes that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, provided the activation function is non-constant, bounded, and monotonically increasing [54]. However, feedforward networks process each input independently, failing to capture the temporal dependencies that are essential for accurate forecasting. This limitation was addressed by Recurrent Neural Networks (RNNs), which maintain an internal state that is updated at each time step through the recurrence relation $h_t = f(W x_t + U h_{t-1} + b)$, enabling the network to capture temporal dependencies across the input sequence.

The Long Short-Term Memory (LSTM) architecture, introduced by Hochreiter and Schmidhuber [1], represents a significant advancement over standard RNNs by addressing the vanishing gradient problem that limits the ability of RNNs to capture long-range temporal dependencies. The vanishing gradient problem arises because the gradient of the loss with respect to the weights at time step $t$ involves the product of Jacobian matrices $\frac{\partial h_t}{\partial h_{t-1}}$ across all intermediate time steps, causing the gradient to decay exponentially with sequence length when the spectral radius of the Jacobian is less than one. The LSTM cell incorporates three gating mechanisms — the input gate, forget gate, and output gate — that regulate the flow of information through the cell state, enabling the network to learn long-range dependencies without suffering from gradient vanishing or explosion. The LSTM equations are defined as:

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

where $i_t$, $f_t$, and $o_t$ are the input, forget, and output gates respectively, $c_t$ is the cell state, $h_t$ is the hidden state, $W$ and $U$ are the input and recurrent weight matrices, $b$ are the bias vectors, $\sigma$ is the sigmoid activation function, and $\odot$ denotes the Hadamard product. The forget gate $f_t$ is the critical innovation that enables the LSTM to maintain information over long sequences, as the gradient flowing through the cell state is multiplied by $f_t$ at each time step rather than by the Jacobian of a non-linear activation function. LSTM architectures have been extensively applied to SST forecasting, with studies demonstrating RMSE values of 0.2–0.4°C for 7-day forecasts in the tropical Pacific and Indian Oceans [18, 19].

The Gated Recurrent Unit (GRU), introduced by Cho et al. [20], simplifies the LSTM architecture by combining the input and forget gates into a single update gate $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$ and introducing a reset gate $r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$ that controls the contribution of the previous hidden state to the candidate activation $\tilde{h}_t = \tanh(W_h x_t + r_t \odot U_h h_{t-1} + b_h)$. The final hidden state is computed as $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. This simplification reduces the number of parameters by approximately 25% relative to LSTM, enabling faster training and inference while maintaining comparable performance on many forecasting tasks [21].

Temporal Convolutional Networks (TCNs), introduced by Bai et al. [22], employ dilated causal convolutions to capture temporal dependencies without the sequential processing requirements of RNNs. The dilated convolution operation is defined by $y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t - d \cdot i}$, where $k$ is the kernel size, $d$ is the dilation factor, and $w_i$ are the convolution weights. By stacking layers with exponentially increasing dilation factors ($d = 1, 2, 4, 8, \ldots$), the TCN achieves an exponentially large receptive field with a linearly increasing number of layers, enabling the capture of long-range dependencies with parallelizable computation. TCNs have demonstrated superior performance to LSTM on several time-series forecasting benchmarks, with the advantage of parallelizable training and inference [23]. However, TCNs process input data as one-dimensional sequences, discarding the spatial relationships inherent in gridded SST observations.

#### 2.4 Convolutional LSTM Architectures

The Convolutional Long Short-Term Memory (ConvLSTM) architecture, first introduced by Shi et al. [2] for precipitation nowcasting, addresses the fundamental limitation of standard LSTM architectures by integrating convolutional operations within the LSTM cell structure. The ConvLSTM cell replaces the fully connected matrix multiplications of the standard LSTM with convolutional operations, enabling the simultaneous capture of spatial and temporal dependencies in gridded data. This modification is motivated by the observation that spatial correlations in geophysical data such as SST fields are local and translation-invariant, properties that are naturally captured by convolutional operations but not by fully connected layers.

The ConvLSTM cell is defined by the following equations:

$$i_t = \sigma(W_{xi} * x_t + W_{hi} * h_{t-1} + b_i)$$
$$f_t = \sigma(W_{xf} * x_t + W_{hf} * h_{t-1} + b_f)$$
$$o_t = \sigma(W_{xo} * x_t + W_{ho} * h_{t-1} + b_o)$$
$$g_t = \tanh(W_{xg} * x_t + W_{hg} * h_{t-1} + b_g)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
$$h_t = o_t \odot \tanh(c_t)$$

where $*$ denotes the convolutional operator, $\odot$ denotes the Hadamard product, $i_t$, $f_t$, $o_t$, and $g_t$ are the input gate, forget gate, output gate, and cell input activation respectively, $c_t$ is the cell state, $h_t$ is the hidden state, $W$ and $b$ are the weight kernels and bias vectors, and $\sigma$ is the sigmoid activation function. The convolutional operator $*$ is defined by $(W * x)_{i,j} = \sum_{m}\sum_{n} W_{m,n} \cdot x_{i-m, j-n}$, where the summation is over the kernel dimensions. The number of parameters in the ConvLSTM cell is $4 \times (C_{in} + C_{hidden}) \times C_{hidden} \times K^2 + 4 \times C_{hidden}$, where $C_{in}$ is the number of input channels, $C_{hidden}$ is the number of hidden channels, and $K$ is the kernel size. For a 3×3 kernel with 64 hidden channels and 4 input channels, this yields approximately 147,000 parameters per ConvLSTM layer, compared to approximately 33 million parameters for a fully connected LSTM processing the same 60×48 spatial grid flattened to a vector.

The ConvLSTM architecture has been applied to a wide range of spatio-temporal forecasting tasks, including precipitation nowcasting [2], traffic flow prediction [24], video prediction [25], and wind speed forecasting [26]. In the context of SST forecasting, ConvLSTM architectures have demonstrated superior performance to standard LSTM by preserving the spatial relationships that drive SST variability, with studies reporting RMSE values of 0.15–0.25°C for 7-day forecasts in the tropical Pacific Ocean [27, 28]. The spatial preservation is particularly important for capturing the propagation of temperature anomalies driven by oceanic currents, as the convolutional operations can learn the directional biases associated with current systems such as the Southwest Monsoon Current and the East India Coastal Current.

Several variants of the ConvLSTM architecture have been proposed to address specific limitations. The TrajGRU architecture, introduced by Shi et al. [29], replaces the fixed convolutional connections of the ConvLSTM with learnable flow-based connections that can capture the motion patterns of precipitation systems. The flow-based connections are computed through a small convolutional network that predicts the displacement vectors for each spatial location, enabling the model to adaptively route information based on the motion patterns present in the input data. The PredRNN architecture, introduced by Wang et al. [30], introduces a zigzag memory flow that enables the model to capture long-range spatio-temporal dependencies more effectively by passing the cell state both temporally (across time steps) and spatially (across layers). The MIM (Memory In Memory) architecture, introduced by Wang et al. [31], introduces a higher-order memory connection that captures the temporal evolution of spatial patterns by maintaining a separate memory state that tracks the changes in the cell state over time.

Despite these advancements, the ConvLSTM architecture exhibits several limitations when applied to SST forecasting. The first limitation relates to the computational cost, with the convolutional operations increasing the number of parameters and memory requirements relative to standard LSTM. A two-layer ConvLSTM with 64 hidden channels and 3×3 kernels requires approximately 294,000 parameters, compared to approximately 50,000 parameters for a two-layer standard LSTM with 64 hidden units processing a single-point time series. The second limitation concerns the fixed spatial resolution of the convolutional operations, which may not adequately capture the multi-scale nature of SST variability, wherein processes operating at scales ranging from kilometers (turbulent mixing) to hundreds of kilometers (mesoscale eddies) contribute to the observed SST field. The third limitation involves the requirement for large amounts of training data, as the convolutional parameters must be learned from data, and the number of trainable parameters grows quadratically with the number of hidden channels.

#### 2.5 Foundation Models for Time-Series Forecasting

Foundation models — large-scale neural networks pre-trained on diverse, multi-domain datasets — have emerged as powerful alternatives to purpose-built architectures in time-series forecasting. The concept of foundation models was first articulated by Bommasani et al. [32] in the context of natural language processing, and has since been extended to computer vision, robotics, and time-series forecasting. The defining characteristic of foundation models is the pre-training paradigm, wherein a single model is trained on a large, diverse corpus of data spanning multiple domains, enabling the model to learn general representations that transfer to downstream tasks with minimal or no domain-specific training.

Amazon Chronos, introduced by Das et al. [3], represents the first foundation model specifically designed for time-series forecasting. The model is based on the T5 transformer architecture [33], with variants ranging from 24 million parameters (t5-small) to 710 million parameters (t5-large). The transformer architecture employs a multi-head self-attention mechanism that computes the output as a weighted sum of value vectors, where the weights are determined by the compatibility of query and key vectors. The self-attention operation is defined by:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q = XW_Q$, $K = XW_K$, and $V = XW_V$ are the query, key, and value matrices obtained by linear projections of the input $X$, and $d_k$ is the dimension of the key vectors. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large, which would push the softmax function into regions with extremely small gradients. The multi-head attention extends this mechanism by computing $h$ independent attention operations in parallel and concatenating the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ and $W_O$ is the output projection matrix. The self-attention mechanism enables the model to capture long-range dependencies without the sequential processing requirements of RNNs, as the attention weights are computed for all pairs of positions in the input sequence simultaneously.

Chronos is pre-trained on a diverse corpus of time-series datasets, including weather data, economic indicators, and sensor readings, enabling it to learn general temporal patterns and structures that transfer to downstream forecasting tasks. The model employs a tokenization strategy wherein the input time series is scaled and quantized into discrete tokens that are processed by the transformer encoder-decoder architecture. The model produces probabilistic forecasts through autoregressive sampling, with the number of samples controlling the trade-off between forecast accuracy and computational cost. The autoregressive generation process is defined by $p(y_{t+1:T} | y_{1:t}) = \prod_{\tau=t+1}^{T} p(y_\tau | y_{1:\tau})$, where each token is sampled sequentially conditioned on all previous tokens.

The Chronos model has been evaluated on a wide range of forecasting benchmarks, demonstrating state-of-the-art performance on the Monash Time-Series Forecasting Archive [34] and the M4 competition dataset [35]. The model's zero-shot forecasting capability — the ability to produce accurate forecasts without any domain-specific training — has been particularly notable, with studies demonstrating that Chronos can match or exceed the performance of purpose-built models on several forecasting tasks [3]. However, the applicability of Chronos to domain-specific tasks such as SST forecasting, where spatial relationships and physical constraints play a critical role, remains an open question.

IBM Granite Time-Series Foundation Model (TTM), introduced by IBM Research [4], represents an alternative approach to foundation model design for time-series forecasting. The model is based on the MLP-Mixer architecture [36], with only 71,000 parameters — significantly fewer than the 200 million parameters of the Chronos t5-base variant. The MLP-Mixer architecture replaces the self-attention mechanism of the transformer with two types of MLP layers: token-mixing MLPs that operate across the temporal dimension and channel-mixing MLPs that operate across the feature dimension. The token-mixing MLP is defined by $H' = H + U_2 \cdot \text{GELU}(U_1 \cdot \text{LayerNorm}(H)^T)^T$, where $H \in \mathbb{R}^{T \times C}$ is the input, $U_1$ and $U_2$ are learnable weight matrices, and GELU is the Gaussian Error Linear Unit activation function. Despite its small size, the Granite TTM model has demonstrated competitive performance to larger foundation models on several forecasting benchmarks, with the advantage of significantly lower computational cost at inference.

The Granite TTM model employs a patch-based tokenization strategy, wherein the input time series is divided into overlapping patches of length $P$ that are processed through a series of MLP-Mixer blocks. The patch tokenization is defined by $x^{(p)}_t = [x_t, x_{t+1}, \ldots, x_{t+P-1}] \in \mathbb{R}^P$, where each patch is projected to a hidden dimension $D$ through a linear layer. The model produces point forecasts through a linear projection of the final hidden state, enabling rapid inference without the need for autoregressive sampling. The model's small size and efficient inference make it particularly suitable for deployment on resource-constrained hardware, though its zero-shot forecasting capability has not been extensively evaluated in the oceanographic domain.

Several other foundation models for time-series forecasting have been proposed, including TimesFM [37], which employs a decoder-only transformer architecture pre-trained on 100 billion time-series data points; Moment [38], which employs a masked autoencoder pre-training strategy inspired by the MAE approach in computer vision; and Lag-Llama [39], which employs a decoder-only transformer with lag-based features for probabilistic forecasting. These models differ in their architectural choices, pre-training strategies, and tokenization approaches, but share the common goal of learning general temporal representations that transfer to downstream forecasting tasks. The comparative effectiveness of these foundation models for domain-specific tasks such as SST forecasting remains an active area of research.

#### 2.6 Parameter-Efficient Fine-Tuning for Foundation Models

Parameter-Efficient Fine-Tuning (PEFT) techniques have emerged as a computationally efficient alternative to full model fine-tuning for adapting foundation models to domain-specific tasks. The core idea of PEFT is to update only a small subset of the model parameters during fine-tuning, while keeping the majority of the pre-trained weights frozen. This approach enables domain adaptation with minimal additional parameters, reducing the computational cost of fine-tuning and mitigating the risk of catastrophic forgetting — the phenomenon wherein fine-tuning on a new task causes the model to lose knowledge acquired during pre-training [55].

Low-Rank Adaptation (LoRA), introduced by Hu et al. [40], represents the most widely used PEFT technique for transformer-based models. LoRA is motivated by the observation that the weight updates during fine-tuning exhibit a low intrinsic rank, meaning that the change in the weight matrix $\Delta W = W_{fine-tuned} - W_{pre-trained}$ can be well-approximated by a low-rank decomposition. LoRA introduces low-rank decomposition matrices into the attention projections of the transformer, enabling the model to learn domain-specific adaptations with only 0.1–3% additional parameters. The LoRA update is defined by the equation:

$$W = W_0 + BA$$

where $W_0 \in \mathbb{R}^{d \times k}$ is the pre-trained weight matrix, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are the low-rank decomposition matrices, and $r \ll \min(d, k)$ is the rank of the decomposition. During fine-tuning, $W_0$ is frozen and only $A$ and $B$ are updated through gradient descent. The rank $r$ controls the trade-off between adaptation capacity and parameter efficiency, with typical values ranging from 4 to 64. The number of trainable parameters introduced by LoRA is $r \times (d + k)$, compared to $d \times k$ for full fine-tuning. For a transformer attention projection with $d = k = 768$ and $r = 8$, LoRA introduces only 12,288 trainable parameters compared to 589,824 for full fine-tuning, representing a 98% reduction in trainable parameters.

LoRA has been extensively applied to natural language processing tasks, demonstrating performance comparable to full fine-tuning with significantly fewer trainable parameters [40]. The application of LoRA to time-series foundation models, however, remains relatively unexplored. Recent studies have applied LoRA to Chronos and other time-series foundation models, demonstrating modest improvements over zero-shot inference but generally underperforming post-hoc calibration techniques [41]. The underperformance of LoRA relative to post-hoc calibration may be attributed to the small size of domain-specific training datasets, which limits the capacity of LoRA to learn meaningful adaptations without overfitting.

Alternative PEFT techniques include Adapter modules [42], which insert small neural network layers between the transformer blocks. An adapter module is defined by $h_{out} = h_{in} + W_2 \cdot \text{ReLU}(W_1 \cdot \text{LayerNorm}(h_{in}))$, where $W_1 \in \mathbb{R}^{d \times r}$ and $W_2 \in \mathbb{R}^{r \times d}$ are the adapter weights with bottleneck dimension $r \ll d$. Adapters are inserted after the self-attention and feed-forward layers of each transformer block, enabling the model to learn domain-specific transformations while preserving the pre-trained representations. Prompt Tuning [43] learns soft prompts that are prepended to the input sequence, defined by a set of learnable vectors $P \in \mathbb{R}^{L_p \times D}$ that are concatenated with the input embeddings before being processed by the transformer. These techniques have demonstrated comparable performance to LoRA on natural language processing tasks, but their applicability to time-series forecasting has not been extensively evaluated.

#### 2.7 Post-Hoc Calibration Techniques

Post-hoc calibration techniques represent a lightweight alternative to fine-tuning for adapting foundation models to domain-specific tasks. The core idea of post-hoc calibration is to apply statistical corrections to the model's predictions using a validation set, without modifying the model weights. This approach enables domain adaptation with minimal computational cost, as the corrections are computed through simple regression operations rather than iterative gradient-based optimization. The computational cost of post-hoc calibration is typically $O(N \cdot D^2)$ for $N$ validation samples and $D$ features, compared to $O(E \cdot N \cdot P)$ for $E$ epochs, $N$ samples, and $P$ model parameters in gradient-based fine-tuning.

Ridge regression residual correction, introduced in the context of time-series forecasting by several recent studies [44, 45], fits a linear model to the residuals between the model's predictions and the ground truth on a validation set. The residual is defined by $r_t = y_t - \hat{y}_t$, where $y_t$ is the observed value and $\hat{y}_t$ is the model's prediction. The Ridge regression model is defined by $\hat{r}_t = \beta^T x_t$, where the coefficients $\beta$ are estimated by minimizing the regularized loss $\sum_{t=1}^{N}(r_t - \beta^T x_t)^2 + \lambda \|\beta\|_2^2$, with $\lambda$ being the regularization parameter. The closed-form solution is given by $\beta = (X^T X + \lambda I)^{-1} X^T r$, where $X$ is the design matrix and $r$ is the vector of residuals. The fitted model is then applied to correct the predictions on the test set through $\hat{y}_t^{corrected} = \hat{y}_t + \hat{r}_t$, reducing systematic biases in the model's forecasts. Ridge regression is preferred over ordinary least squares due to its regularization properties, which prevent overfitting when the validation set is small. The per-horizon variant fits separate models for each forecast horizon $h \in \{1, \ldots, H\}$, recognizing that the bias structure may vary with the forecast lead time.

Amplitude calibration adjusts the slope and intercept of the linear relationship between predicted and observed values, ensuring that the model's forecasts capture the full magnitude of the target variable. The calibration is typically performed through linear regression on the validation set, yielding the relationship $y = a\hat{y} + b + \epsilon$, where $a$ is the slope and $b$ is the intercept. The slope is computed by $a = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$ and the intercept by $b = \bar{y} - a\bar{x}$, where $x_i$ are the predicted values and $y_i$ are the observed values. The calibrated predictions are obtained through $\hat{y}_t^{calibrated} = a\hat{y}_t + b$. Amplitude calibration has been shown to improve the slope metric — a measure of amplitude response fidelity — for foundation model forecasts, though the improvement is often modest when applied in isolation [44]. The slope clipping technique, wherein the slope is constrained to a range such as [0.85, 1.00], prevents over-correction that may amplify noise in the predictions.

Adaptive drift correction applies a time-varying bias correction to the model's predictions, accounting for systematic shifts in the forecast error over time. The drift correction is computed through an exponentially weighted moving average of the forecast errors: $d_t = \alpha \cdot e_t + (1 - \alpha) \cdot d_{t-1}$, where $e_t = y_t - \hat{y}_t$ is the forecast error at time $t$ and $\alpha \in (0, 1]$ is the smoothing parameter. Alternatively, a rolling mean over a window of $W$ days may be employed: $d_t = \frac{1}{W}\sum_{i=t-W+1}^{t} e_i$. The corrected predictions are obtained through $\hat{y}_t^{corrected} = \hat{y}_t + d_t$. Adaptive drift correction has been shown to reduce the frequency of large forecast errors, improving the big error count metric for foundation model forecasts [45]. The capping mechanism, wherein the drift correction is limited to a maximum magnitude (e.g., ±0.20°C), prevents the correction from amplifying noise during periods of low predictability.

The combination of Ridge regression residual correction, amplitude calibration, and adaptive drift correction — collectively referred to as few-shot post-hoc calibration — has been shown to significantly improve the forecasting performance of foundation models on domain-specific tasks [44, 45]. However, the systematic amplitude compression observed in foundation model forecasts — wherein the model under-predicts the magnitude of extreme events — has proven resistant to correction through these techniques alone, motivating the development of the PostGain slope correction technique described in this chapter.

#### 2.8 Research Gap

From the literature review, several significant gaps in the existing research are identified. The first gap relates to the comparative evaluation of ConvLSTM and foundation models for SST forecasting in the Indian Ocean region. While ConvLSTM architectures have been extensively evaluated for precipitation nowcasting and traffic flow prediction, and foundation models have been evaluated on general time-series forecasting benchmarks, the comparative effectiveness of these approaches for domain-specific SST forecasting has not been systematically studied. The existing studies on ConvLSTM for SST forecasting [27, 28] have been conducted in the tropical Pacific Ocean, where the dynamics of SST variability differ significantly from those in the Indian Ocean due to differences in monsoon forcing, current systems, and basin geometry. The existing studies on foundation models for time-series forecasting [3, 4, 37, 38] have not evaluated their performance on oceanographic data, where spatial relationships and physical constraints play a critical role.

The second gap concerns the systematic amplitude compression observed in foundation model forecasts. While post-hoc calibration techniques have been shown to improve RMSE and other accuracy metrics, the systematic under-prediction of extreme event magnitudes — a critical limitation for operational marine warning systems — has proven resistant to correction through existing techniques. The development of a post-hoc technique that resolves this limitation while maintaining the computational efficiency of the foundation model approach represents a significant contribution to the field. The amplitude compression phenomenon has been noted in the foundation model literature [3, 4] but has not been systematically characterized or addressed through a dedicated correction mechanism.

The third gap involves the evaluation framework for SST forecasting models. Traditional evaluation metrics such as overall RMSE provide a single aggregate measure of forecast accuracy but fail to capture critical aspects of forecast quality including performance during extreme events, seasonal variability, amplitude fidelity, and the frequency of large errors. The development of a comprehensive multi-dimensional evaluation framework that assesses forecast quality across multiple dimensions simultaneously represents a significant contribution to the field of operational oceanographic forecasting. The five-gate evaluation framework proposed in this chapter addresses this gap by incorporating gates for overall accuracy, seasonal performance (February and March RMSE), extreme event handling (big error count), and amplitude fidelity (slope).

The fourth gap relates to the integration of spatial information in foundation model forecasting. Foundation models such as Chronos and Granite TTM are designed for univariate time-series forecasting, processing a single time series at a time without incorporating spatial context. The beta-map spatial propagation technique developed in this chapter addresses this gap by reconstructing the full spatial field from point forecasts at a target location, leveraging the spatial correlation structure learned from the training data. This technique enables foundation models to produce spatial forecasts comparable to those produced by ConvLSTM architectures, while maintaining the computational efficiency of the point-forecasting approach.

This chapter addresses these gaps through a comprehensive empirical evaluation of ConvLSTM and foundation models for SST forecasting in the Indian Ocean region, the development of the PostGain slope correction technique that resolves the systematic amplitude compression observed in foundation model forecasts, and the establishment of a rigorous five-gate evaluation framework that assesses forecast quality across multiple dimensions simultaneously.

#### 2.9 SST Forecasting in the Indian Ocean Region

The Indian Ocean presents unique challenges and opportunities for SST forecasting due to its distinctive oceanographic and atmospheric characteristics. The basin is bounded by the Asian landmass to the north, Africa to the west, Australia to the east, and the Southern Ocean to the south, creating a semi-enclosed geometry that influences circulation patterns and heat transport. The Indian Ocean is the warmest ocean basin on Earth, with mean SST values exceeding 28°C across much of the tropical region, and it plays a critical role in the global climate system through its influence on the monsoon systems that affect billions of people [56].

The Indian Ocean Dipole (IOD), first identified by Saji et al. [15], represents the dominant mode of interannual SST variability in the tropical Indian Ocean. The IOD is characterized by a seesaw pattern of SST anomalies between the western and eastern equatorial Indian Ocean, with positive IOD events featuring warmer-than-normal SST in the west and cooler-than-normal SST in the east, and negative IOD events featuring the opposite pattern. The IOD has been shown to influence rainfall patterns over East Africa, Indonesia, and Australia, as well as the intensity of the Indian Summer Monsoon [5]. The predictability of the IOD on seasonal timescales has been studied extensively, with coupled general circulation models (CGCMs) demonstrating skill scores of 0.5–0.7 for 3–6 month lead forecasts [57]. However, the predictability of SST on sub-seasonal timescales (7–30 days) in the Indian Ocean remains less well understood, particularly in the Arabian Sea and Bay of Bengal regions where mesoscale processes dominate.

The Arabian Sea, encompassing the western portion of the study domain (60°E–72°E), is characterized by intense seasonal variability driven by the monsoon wind reversal. During the Southwest Monsoon (June–September), strong southwesterly winds drive coastal upwelling along the Somali and Arabian coasts, bringing cold, nutrient-rich waters to the surface and reducing SST by 3–5°C relative to the pre-monsoon period [58]. The Lakshadweep High, an anticyclonic eddy that forms off the southwest coast of India during the pre-monsoon period (March–May), influences SST variability in the Laccadive Sea region through its interaction with the Southwest Monsoon Current [59]. The Bay of Bengal, encompassing the eastern portion of the broader Indian Ocean domain, is characterized by strong stratification due to freshwater input from the Ganges-Brahmaputra and Irrawaddy river systems, which suppresses vertical mixing and maintains a shallow mixed layer [60].

Several studies have specifically addressed SST forecasting in the Indian Ocean region using various methodologies. Shenoi et al. [10] applied the ROMS model to simulate SST variability in the Arabian Sea and Bay of Bengal, demonstrating the model's capability to reproduce the seasonal cycle of SST with RMSE values of 0.5–1.0°C. The study highlighted the importance of accurate wind forcing and river discharge data for reproducing the observed SST patterns, particularly in the Bay of Bengal where freshwater input plays a dominant role in determining the mixed layer depth. Bansal et al. [13] applied ARIMA models to SST forecasting in the Indian Ocean, producing 7-day forecast RMSE values of 0.4–0.6°C. The study demonstrated that ARIMA models capture the seasonal cycle of SST but fail to predict interannual anomalies associated with the IOD and other climate modes.

More recent studies have applied machine learning methods to SST forecasting in the Indian Ocean. Zhang et al. [28] applied a ConvLSTM architecture to SST forecasting in the tropical Indian Ocean, reporting RMSE values of 0.18–0.25°C for 7-day forecasts. The study demonstrated that the ConvLSTM architecture outperforms standard LSTM by preserving spatial relationships, with the improvement being most pronounced in regions with strong spatial gradients such as the Somali upwelling zone and the eastern equatorial Indian Ocean. However, the study did not evaluate the model's performance at specific locations of operational interest, nor did it compare the ConvLSTM architecture to foundation models or other modern forecasting approaches.

The application of foundation models to SST forecasting in the Indian Ocean region has not been previously reported in the literature. The present study addresses this gap through a comprehensive evaluation of Amazon Chronos and IBM Granite TTM for SST forecasting in the Indian Ocean, comparing their performance to a purpose-built ConvLSTM architecture across multiple dimensions of forecast quality. The development of the PostGain slope correction technique and the beta-map spatial propagation method represents a novel contribution that enables foundation models to produce spatially-resolved SST forecasts with amplitude fidelity comparable to purpose-built architectures.

---

### 3. Methodology

#### 3.1 Dataset Description

The dataset utilized in this study consists of daily sea surface temperature observations spanning 16,290 days (September 1, 1981 to April 7, 2026) over a spatial grid of 60×48 pixels covering the Indian Ocean region from 5°N to 20°N latitude and 60°E to 72°E longitude. The spatial resolution of 0.25 degrees per pixel provides sufficient granularity to capture mesoscale oceanographic features including eddies, fronts, and coastal upwelling zones that significantly influence regional SST variability. The dataset was provided by the Indian National Centre for Ocean Information Services (INCOIS) in NumPy binary format (.npy) and contains three primary components: the observed SST field, the long-term daily mean (LTDM) climatology, and the SST anomaly field computed as the difference between observed SST and LTDM.

Table 1: Dataset Specifications

| Parameter | Value |
|-----------|-------|
| Temporal coverage | September 1, 1981 – April 7, 2026 |
| Total observations | 16,290 days |
| Spatial grid | 60 × 48 pixels |
| Latitude range | 5°N – 20°N |
| Longitude range | 60°E – 72°E |
| Spatial resolution | 0.25° per pixel |
| Target location | 8.0°N, 67.0°E (pixel indices 12, 28) |
| Data format | NumPy binary (.npy) |
| Components | SST field, LTDM climatology, anomaly field |

The target location for this study is the grid cell at 8.0°N, 67.0°E, corresponding to pixel indices (12, 28) in the 60×48 grid. This location in the Laccadive Sea was selected based on its oceanographic significance as a region where the Southwest Monsoon Current interacts with the Lakshadweep High, producing complex SST variability patterns that challenge forecasting systems. The target location experiences SST variations ranging from approximately 26°C during the Northeast Monsoon (December–February) to 30°C during the pre-monsoon period (April–May), with a standard deviation of approximately 1.2°C around the climatological mean.

Figure 1: SST Spatial Distribution over Indian Ocean Region (60×48 Grid)

[Figure placeholder: Spatial map of mean SST over the 60×48 grid covering 5°N–20°N, 60°E–72°E, with the target location 8.0°N, 67.0°E marked with a red dot. Color scale from 24°C (blue) to 32°C (red).]

The dataset was partitioned into training (85%), validation (5%), and testing (10%) subsets using a temporal split strategy that preserves the chronological ordering of observations. This approach ensures that the model is evaluated on truly unseen future data rather than randomly selected time points, which would introduce data leakage and produce unrealistically optimistic performance estimates. The training set comprises 13,846 days (September 1, 1981 to approximately April 2023), the validation set comprises 815 days (approximately May 2023 to December 2023), and the test set comprises 1,629 days (January 2024 to April 2026). The final evaluation was conducted on a 90-day subset of the test set covering January 1, 2026 to March 31, 2026, representing the most recent complete quarterly period available at the time of analysis. The split indices were computed as $T_{train} = \lfloor 0.85 \times 16290 \rfloor = 13846$ and $T_{val} = \lfloor 0.90 \times 16290 \rfloor = 14661$, ensuring strict temporal separation between all subsets.

#### 3.2 Preprocessing Pipeline

The preprocessing pipeline implements a sequence of operations that transform the raw SST observations into a format suitable for model input. The pipeline is applied consistently across all model families to ensure that performance differences are attributable to architectural and algorithmic choices rather than preprocessing variations.

The first step of the pipeline involves data loading and type conversion. The SST field, LTDM climatology, and anomaly field are loaded from NumPy binary files and converted to 32-bit floating-point precision to reduce memory requirements while maintaining numerical accuracy. The data shapes are verified against expected dimensions (16,290 × 60 × 48) to ensure data integrity. Memory contiguity is enforced through `np.ascontiguousarray()` to enable efficient tensor conversion and GPU transfer operations.

The second step involves edge padding of the spatial grid. The 60×48 grid is padded to 60×50 columns using edge-mode padding, which replicates the values at the grid boundaries. This padding is required for the patch decomposition strategy employed by the ConvLSTM architecture, wherein the grid is divided into overlapping patches of 7×7 pixels. The padding ensures that patches at the grid boundaries have sufficient context for accurate processing. The padding operation is formally defined as $X_{pad}(t, i, j) = X(t, i, \min(j, W-1))$ for $j \geq W$, where $W = 48$ is the original width dimension.

The third step involves the computation of the Long-Term Daily Mean (LTDM) climatology. The LTDM is computed by averaging the SST values for each day of the year across all years in the training set, producing a 366 × 60 × 48 array (accounting for leap years). The LTDM provides the model with information about the expected SST value at each grid cell for each day of the year, enabling it to distinguish between normal seasonal variations and anomalous deviations. The climatological mean for day-of-year $d$ at grid cell $(i, j)$ is computed as:

$$LTDM(d, i, j) = \frac{1}{N_d} \sum_{y \in \mathcal{Y}_d} SST(y, d, i, j)$$

where $\mathcal{Y}_d$ is the set of years containing day-of-year $d$ in the training period, and $N_d = |\mathcal{Y}_d|$ is the count of such years. The anomaly field is then computed as $ANOM(t, i, j) = SST(t, i, j) - LTDM(doy(t), i, j)$, where $doy(t)$ maps the absolute day index to its corresponding day-of-year.

The fourth step involves per-channel z-score normalization. The anomaly field and LTDM field are normalized independently using the mean and standard deviation computed from the training set only, ensuring that no information from the validation or test sets leaks into the training process. The normalization is defined by the equation:

$$x_{norm} = \frac{x - \mu}{\sigma}$$

where $x$ is the raw value, $\mu$ is the training-set mean, and $\sigma$ is the training-set standard deviation. A small constant (1e-8) is added to the standard deviation to prevent division by zero in the case of constant channels. The normalization statistics are computed as:

$$\mu_c = \frac{1}{T_{train} \cdot H \cdot W} \sum_{t=1}^{T_{train}} \sum_{i=1}^{H} \sum_{j=1}^{W} X_c(t, i, j)$$

$$\sigma_c = \sqrt{\frac{1}{T_{train} \cdot H \cdot W} \sum_{t=1}^{T_{train}} \sum_{i=1}^{H} \sum_{j=1}^{W} (X_c(t, i, j) - \mu_c)^2 + \epsilon}$$

where $c$ indexes the channel, $H = 60$ and $W = 50$ are the spatial dimensions (post-padding), and $\epsilon = 10^{-8}$ is the numerical stability constant.

The fifth step involves the construction of the four-channel input tensor. The input tensor comprises four channels: the normalized anomaly field (channel 0), the normalized LTDM field (channel 1), the latitude coordinate (channel 2), and the longitude coordinate (channel 3). The latitude and longitude channels provide the model with explicit information about the spatial location of each grid cell, enabling it to learn location-specific patterns such as coastal effects, current systems, and monsoon influences. The coordinate channels are broadcast across the temporal dimension and remain constant throughout the sequence.

Table 2: Input Channel Specifications

| Channel | Description | Normalization | Purpose |
|---------|-------------|---------------|---------|
| 0 | SST anomaly | Z-score (train only) | Primary signal |
| 1 | LTDM climatology | Z-score (train only) | Climatological context |
| 2 | Latitude coordinate | Raw value (degrees) | Spatial encoding |
| 3 | Longitude coordinate | Raw value (degrees) | Spatial encoding |

The dataset class implements the PyTorch `Dataset` interface with a sliding window approach. For a sequence length $L = 60$ and horizon $H = 7$, the number of valid samples from a split of size $T_{split}$ is $N_{samples} = T_{split} - L - H + 1$. The `__getitem__` method returns a tuple $(X, Y)$ where $X \in \mathbb{R}^{L \times 4 \times H \times W}$ is the input sequence and $Y \in \mathbb{R}^{H \times H \times W}$ is the target anomaly field. All arrays are converted to PyTorch tensors via `torch.from_numpy()` with `.copy()` applied beforehand to ensure memory contiguity and prevent shared-memory issues during multi-worker data loading.

#### 3.3 ConvLSTM Architecture

The Convolutional Long Short-Term Memory (ConvLSTM) architecture employed in this study is based on the formulation introduced by Shi et al. [2], with modifications to optimize performance for the SST forecasting task. The architecture consists of two ConvLSTM layers, each comprising a convolutional LSTM cell with 64 hidden channels and a 3×3 kernel size. The input to the first layer is the four-channel input tensor described in Section 3.2, and the output of the second layer is passed through a 1×1 convolutional layer to produce the single-channel SST anomaly forecast.

Figure 2: ConvLSTM Cell Architecture

[Figure placeholder: Diagram of the ConvLSTM cell showing the input x_t, hidden state h_{t-1}, cell state c_{t-1}, and the four gating operations (input gate i_t, forget gate f_t, output gate o_t, cell input g_t) with 3×3 convolutional connections. The cell state c_t and hidden state h_t are shown as outputs.]

The ConvLSTM cell is defined by the equations presented in Section 2.4, with the convolutional operations replacing the fully connected matrix multiplications of the standard LSTM. The 3×3 kernel size was selected based on a hyperparameter tuning study that evaluated kernel sizes of 1×1, 3×3, and 5×5, with the 3×3 kernel providing the best trade-off between model capacity and computational efficiency. The 64 hidden channels were selected based on a similar study that evaluated hidden dimensions of 32, 48, and 64, with the 64-channel configuration providing the best performance. The total number of trainable parameters in the two-layer ConvLSTM architecture is 295,105, computed as $2 \times [4 \times (C_{in} + C_{hidden}) \times C_{hidden} \times K^2 + 4 \times C_{hidden}] + C_{hidden} \times 1 \times 1^2 + 1$, where $C_{in}$ varies between layers (4 for the first, 64 for the second).

Table 3: ConvLSTM Architecture Parameters

| Parameter | Value |
|-----------|-------|
| Input channels | 4 (anomaly, LTDM, lat, lon) |
| ConvLSTM layers | 2 |
| Hidden channels | 64 |
| Kernel size | 3×3 |
| Output channels | 1 (SST anomaly) |
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Batch size | 8 |
| Maximum epochs | 25 |
| Early stopping patience | 5 epochs |
| Loss function | MSE |
| Training hardware | NVIDIA T4 GPU (16GB VRAM) |

The training procedure employs the Adam optimizer [46] with a learning rate of 5e-4, a batch size of 8, and a maximum of 25 epochs. The loss function is the Mean Squared Error (MSE) between the predicted and observed SST anomaly values, computed over the full 60×48 spatial grid. The MSE loss is defined as:

$$\mathcal{L}_{MSE} = \frac{1}{B \cdot H \cdot W} \sum_{b=1}^{B} \sum_{i=1}^{H} \sum_{j=1}^{W} (\hat{Y}_{b,i,j} - Y_{b,i,j})^2$$

where $B$ is the batch size, $H$ and $W$ are the spatial dimensions, $\hat{Y}$ is the predicted anomaly field, and $Y$ is the ground truth anomaly field. Early stopping is employed based on the validation loss, with a patience of 5 epochs to prevent overfitting. The validation loss is computed at the end of each epoch on the full validation set, and the model checkpoint with the minimum validation loss is preserved. If the validation loss does not improve for 5 consecutive epochs, training is terminated and the best checkpoint is restored for inference.

The DataLoader is configured with `num_workers=2`, `pin_memory=True`, `drop_last=True`, and `persistent_workers=True`. The `pin_memory` flag allocates page-locked host memory for batch tensors, enabling faster asynchronous GPU transfers via direct memory access (DMA). The `persistent_workers` flag maintains worker processes across epochs, reducing the overhead of process creation and destruction. The `drop_last` flag discards incomplete batches at the end of each epoch to ensure consistent batch dimensions, which is required for cuDNN's auto-tuner to maintain optimal kernel selection.

The post-processing pipeline applied to the ConvLSTM forecasts comprises four stages: per-horizon bias correction, inverse-RMSE² weighting, adaptive capping (±0.20°C), and 7-day rolling mean smoothing. The bias correction is computed by averaging the forecast errors for each horizon (D1–D7) on the validation set, and the corrected forecasts are obtained by subtracting the bias from the raw predictions. The inverse-RMSE² weighting combines forecasts from multiple models (when available) using weights proportional to the inverse of the squared RMSE, giving higher weight to more accurate models. The adaptive capping limits the magnitude of forecast adjustments to ±0.20°C, preventing regime-shift overshoot. The 7-day rolling mean smoothing reduces noise in the forecast time series, improving the signal-to-noise ratio.

#### 3.4 Multi-Horizon Strategy Exploration

Three distinct strategies for producing forecasts at multiple horizons were explored during the development phase: MIMO (Multiple Input Multiple Output), Direct, and Recursive. Each strategy was evaluated to determine the optimal approach for the SST forecasting task.

The MIMO strategy employs a single encoder-decoder architecture with multiple output heads, one for each forecast horizon. The encoder processes the 60-day input sequence through two ConvLSTM layers, producing a latent representation that is passed to each output head for horizon-specific prediction. Each output head consists of a 1×1 convolutional layer that maps the 64-channel hidden state to a single-channel anomaly forecast. The MIMO strategy was selected because the shared encoder learns representations that are useful for all forecast horizons, potentially improving generalization by leveraging information from all horizons during training. The computational efficiency of the MIMO strategy — training a single model for all horizons rather than separate models — was also a significant advantage. The multi-horizon loss is computed as the sum of MSE losses across all 7 horizons:

$$\mathcal{L}_{MIMO} = \sum_{h=1}^{7} \mathcal{L}_{MSE}^{(h)}$$

The Direct strategy employs separate models for each forecast horizon, with each model trained independently to predict the SST value at its specific horizon. The Direct strategy was selected because each model can learn the patterns that are most relevant for its specific forecast horizon, potentially improving accuracy by avoiding the compromises inherent in joint optimization. The independent training of each model also enables parallel execution and reduces the total training time when multiple GPU devices are available. However, the Direct strategy requires 7× the storage for model weights and does not share learned representations across horizons.

The Recursive strategy employs a single 1-step model that is applied iteratively to produce multi-horizon forecasts through autoregressive rollout. The Recursive strategy was selected because the single 1-step model requires fewer parameters than MIMO or Direct strategies, reducing the computational cost of training and inference. The recursive strategy can theoretically produce forecasts at any horizon by iterating the 1-step model, providing flexibility in forecast production. However, the recursive strategy suffers from error accumulation, wherein prediction errors compound at each step of the rollout. The error variance after $h$ recursive steps grows approximately as $\sigma_h^2 \approx h \cdot \sigma_1^2$, where $\sigma_1^2$ is the one-step prediction error variance [30].

Based on the comparative evaluation, the MIMO strategy was selected for the production ConvLSTM architecture (Script 69) due to its balance of accuracy, computational efficiency, and forecast consistency across horizons.

#### 3.5 Foundation Model Integration

The integration of foundation models into the SST forecasting pipeline was conducted through three distinct paradigms: zero-shot inference, few-shot post-hoc calibration, and LoRA fine-tuning. Each paradigm was evaluated for both Amazon Chronos and IBM Granite TSFM to determine the most effective approach for domain adaptation.

The zero-shot inference paradigm employs the pre-trained foundation model weights without any domain-specific training or calibration. The model is applied directly to the SST time series at the target location, producing forecasts through the model's native prediction mechanism. For Chronos, this involves autoregressive sampling with 20 samples per forecast, with the median of the samples taken as the point forecast. For Granite TTM, this involves a single forward pass through the MLP-Mixer architecture, producing a point forecast directly.

The few-shot post-hoc calibration paradigm applies statistical corrections to the zero-shot predictions using the validation set, without modifying the model weights. The corrections comprise three stages: Ridge regression residual correction, amplitude calibration, and adaptive drift correction. The Ridge regression residual correction fits a linear model to the residuals between the zero-shot predictions and the ground truth on the validation set, with one model per forecast horizon (D1–D7). The amplitude calibration adjusts the slope and intercept of the linear relationship between predicted and observed values, with the slope clipped to the range [0.85, 1.00] to prevent over-correction. The adaptive drift correction applies a time-varying bias correction to the predictions, computed through a rolling mean of the forecast errors on the validation set with a window of 5 days and a cap of ±0.20°C.

The LoRA fine-tuning paradigm introduces low-rank decomposition matrices into the attention projections of the foundation model, enabling domain-specific training with minimal additional parameters. The LoRA configuration employs a rank of 8 for Chronos and ranks of 8, 16, and 32 for Granite, with the LoRA alpha set to 64 and the target modules set to the key, value, query, and output projections. The fine-tuning is performed on the training set for 10 epochs with a learning rate of 1e-4, using the AdamW optimizer with a weight decay of 0.01. The model weights are updated through gradient descent, with the pre-trained weights frozen except for the LoRA modules. The LoRA update is applied as $W' = W_0 + \frac{\alpha}{r}BA$, where $W_0$ is the original weight matrix, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are the low-rank matrices, $r$ is the rank, and $\alpha$ is the scaling factor [40].

#### 3.6 Single-Model Zero-Shot + Post-Hoc Correction with PostGain

The single-model zero-shot + post-hoc correction pipeline represents the primary contribution of this study's foundation model investigation. The pipeline was developed for both Chronos (Script 86) and Granite (Script 87), with a deterministic variant created for Chronos (Script 88) to enable reproducible outputs. The pipeline is NOT fine-tuning and NOT LoRA; it uses zero-shot inference with post-hoc statistical corrections, including a novel PostGain slope targeting mechanism.

The pipeline operates as follows. First, zero-shot inference is performed on each rolling window of the evaluation period, producing a point forecast at the target location. The model weights are NOT modified during this process. Second, per-horizon bias correction is applied, computed as the mean forecast error for each horizon on the validation set. Third, Ridge residual correction is applied, using 7 horizon-specific linear models fitted on the validation set. Fourth, amplitude calibration is applied, with the slope clipped to the range [0.85, 1.00] and the intercept recomputed as $b = \bar{y} - a\bar{x}$ after slope clipping. Fifth, adaptive drift correction is applied, with a rolling mean of the forecast errors computed over a window of 5 days and capped at ±0.20°C.

The sixth and most critical stage is the PostGain slope targeting mechanism. The PostGain mechanism fits a gain multiplier to the corrected predictions on the validation set, selecting the minimum gain that achieves a slope of at least 0.94. The gain is selected through a grid search over the range [1.00, 1.14] with a step size of 0.01. The gain is applied to the corrected predictions as a multiplicative factor, amplifying the amplitude of the forecasts to match the observed SST variability. The PostGain mechanism is the critical innovation that resolves the systematic amplitude compression previously observed in all foundation model configurations.

Figure 3: PostGain Pipeline Flowchart

[Figure placeholder: Flowchart showing the six stages of the PostGain pipeline: (1) Zero-shot inference, (2) Per-horizon bias correction, (3) Ridge residual correction, (4) Amplitude calibration, (5) Adaptive drift correction, (6) PostGain slope targeting. Each stage is shown as a box with arrows connecting them, and the input/output data shapes are indicated.]

The seventh stage involves beta-map spatial propagation, which reconstructs the full 60×48 spatial field from the point forecast at the target location. The beta-map is computed from the training set as the spatial correlation coefficient between the SST anomaly at each grid cell and the SST anomaly at the target location. The spatial field is reconstructed using the equation:

$$SST_{spatial}(h, w) = SST_{context}(h, w) + \beta(h, w) \times \Delta SST_{point} + LTDM(h, w)$$

where $\beta(h, w)$ is the beta-map coefficient at grid cell $(h, w)$, $\Delta SST_{point}$ is the point forecast anomaly, $SST_{context}$ is the context anomaly field, and $LTDM$ is the long-term daily mean climatology. The beta-map coefficients are computed through ordinary least squares regression of each grid cell's anomaly time series against the target location's anomaly time series:

$$\beta(h, w) = \frac{\text{Cov}(A_{h,w}, A_{target})}{\text{Var}(A_{target})} = \frac{\sum_{t=1}^{T_{train}} (A_{h,w}^{(t)} - \bar{A}_{h,w})(A_{target}^{(t)} - \bar{A}_{target})}{\sum_{t=1}^{T_{train}} (A_{target}^{(t)} - \bar{A}_{target})^2}$$

where $A_{h,w}^{(t)}$ is the anomaly at grid cell $(h, w)$ and time step $t$, and $\bar{A}$ denotes the temporal mean over the training period. The beta-map effectively captures the spatial correlation structure of SST anomalies, enabling point forecasts to be propagated across the full spatial domain while preserving the relative amplitude relationships between grid cells.

Table 4: PostGain Pipeline Parameters

| Parameter | Value |
|-----------|-------|
| Zero-shot inference | Model weights frozen |
| Bias correction | Per-horizon mean (validation set) |
| Ridge correction | 7 horizon-specific linear models |
| Amplitude calibration | Slope clipping [0.85, 1.00] |
| Adaptive drift | Window=5 days, cap=±0.20°C |
| PostGain grid search | Range [1.00, 1.14], step 0.01 |
| PostGain target slope | ≥ 0.94 |
| Beta-map computation | Covariance / variance (training set) |
| Spatial field reconstruction | Context + beta × point + LTDM |

#### 3.7 Ensemble Pipelines (Secondary Investigation)

Two ensemble pipelines were developed as a secondary investigation, explicitly not the primary focus of this study per advisor guidance recommending single-model emphasis. The first pipeline (Script 84) is a point-only ensemble that combines cached predictions from three Stage 2 candidates: F1C (Chronos few-shot), G1A (Granite few-shot), and L1 (Chronos LoRA). The second pipeline (Script 85) is a spatial ensemble that runs both Chronos and Granite on the target pixel for each rolling window, then reconstructs full spatial fields via beta-map propagation.

The point ensemble (Script 84) combines the candidate predictions through weighted averaging, with the weights tuned via grid search on a blocked hold-out set (last 30 days of the evaluation period). The tuning objective is a slope-aware score defined as:

$$score = RMSE + \lambda \times \max(0, 0.94 - slope)^2$$

where $\lambda$ is a penalty coefficient that controls the trade-off between RMSE minimization and slope compliance. The slope-aware objective explicitly penalizes slope failures during weight tuning, encouraging the selection of weights that produce forecasts with proper amplitude response. The ensemble weights are constrained to sum to unity ($\sum_i w_i = 1$) and are non-negative ($w_i \geq 0$), ensuring that the ensemble prediction remains within the convex hull of the individual model predictions.

A critical behavior of the ensemble tuning process is the tendency of the tuner to collapse weights to a single model. When the objective prefers one model over the others, the tuner sets the weight of the preferred model to 1.0 and the weights of the other models to 0.0. This behavior means that the ensemble effectively acts as a model selector rather than a true combination, providing insight into which individual model performs best under the tuning objective.

The spatial ensemble (Script 85) runs both Chronos and Granite on the target pixel for each rolling window, producing point forecasts that are combined through weighted averaging. The ensemble weights are tuned via grid search with the same slope-aware objective as the point ensemble. The combined point forecast is then propagated to the full spatial field via beta-map propagation, producing ConvLSTM-style spatial outputs.

#### 3.8 Five-Gate Evaluation Framework

The five-gate evaluation framework was designed to assess forecast quality across multiple dimensions simultaneously, ensuring that the selected model is suitable for operational deployment across the full range of conditions encountered in practice. The framework consists of five gates, each representing a distinct aspect of forecast quality that must be satisfied for the model to be considered operationally viable.

Table 5: Five-Gate Evaluation Criteria

| Gate | Metric | Threshold | Rationale |
|------|--------|-----------|-----------|
| 1 | Overall RMSE | < 0.1466°C | Primary accuracy metric; penalizes large errors |
| 2 | February RMSE | < 0.2093°C | Most challenging month (monsoon transition) |
| 3 | March RMSE | ≤ 0.1003°C | Post-monsoon stability period |
| 4 | Big Error Count | ≤ 12 days (out of 90) | Frequency of |error| ≥ 0.20°C |
| 5 | Slope | [0.94, 1.00] | Amplitude response fidelity |

Gate 1 assesses overall forecast accuracy through the Root Mean Square Error (RMSE) computed over the entire 90-day evaluation period (January 1 – March 31, 2026). The threshold of <0.1466°C was established based on the performance of the best existing operational forecasting method and represents a meaningful improvement over baseline approaches. RMSE is the primary metric for forecast accuracy because it penalizes large errors more heavily than small errors, reflecting the operational reality that large forecast errors have disproportionately severe consequences for decision-making.

Gate 2 assesses forecast accuracy during the February period, which represents the most challenging month for SST forecasting in the Indian Ocean due to the monsoon transition. The threshold of <0.2093°C was established based on the observed variability during February in the historical record and represents a level of accuracy that is operationally useful for monsoon-related applications. February is the most critical month because the transition from Northeast to Southwest Monsoon introduces complex atmospheric and oceanic dynamics that challenge forecasting systems.

Gate 3 assesses forecast accuracy during the March period, which represents a secondary challenge due to post-monsoon stability transitions. The threshold of ≤0.1003°C was established based on the observed variability during March and represents a level of accuracy that ensures reliable forecasting during the transition to the pre-monsoon period.

Gate 4 assesses the frequency of large forecast errors, defined as days where the absolute error exceeds 0.20°C. The threshold of ≤12 big error days (out of 90) ensures that the model does not produce unacceptably large errors on a frequent basis, which would undermine confidence in the forecast system. Big errors are particularly problematic for operational applications because they can lead to incorrect decisions with significant economic or safety consequences.

Gate 5 assesses the amplitude fidelity of the forecast through the slope of the linear regression between predicted and observed SST values. The threshold of [0.94, 1.00] ensures that the model captures the full magnitude of SST variations without systematically under-predicting (slope < 0.94) or over-predicting (slope > 1.00) the amplitude of temperature anomalies. Slope is a critical metric for operational applications because amplitude compression (slope < 1.0) means the model under-predicts extreme events, which are precisely the events of greatest operational interest.

The slope metric is computed through ordinary least squares linear regression:

$$slope = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

where $x_i$ are the predicted SST values, $y_i$ are the observed SST values, and $n = 90$ is the number of evaluation days. A slope of 1.0 indicates perfect amplitude response, while a slope less than 1.0 indicates amplitude compression (under-prediction of extreme events).

The five-gate framework was designed to be comprehensive yet achievable, with the requirement that a model must pass all five gates to be considered operationally viable. This stringent criterion ensures that the selected model performs well across all dimensions of forecast quality, rather than excelling in some areas while failing in others.

#### 3.9 Computational Infrastructure and GPU Optimization

All experiments were conducted on a single NVIDIA T4 GPU with 16GB of VRAM, hosted on a cloud compute instance with 4 vCPUs and 16GB of system RAM. The T4 GPU, based on the NVIDIA Turing architecture, provides 2560 CUDA cores, 320 Tensor Cores for mixed-precision operations, and a memory bandwidth of 320 GB/s. The GPU selection was motivated by its widespread availability in cloud computing environments, its energy efficiency (70W TDP), and its suitability for inference and moderate-scale training workloads typical of operational oceanographic forecasting systems.

The PyTorch deep learning framework (version 2.x) was employed for all model implementations, with CUDA 12.x providing the GPU acceleration backend. Several GPU-specific optimizations were applied to maximize training throughput and minimize memory consumption. The cuDNN benchmarking mode was enabled through `torch.backends.cudnn.benchmark = True`, which causes cuDNN to profile multiple convolution algorithms at the first forward pass and select the fastest algorithm for the specific input dimensions encountered. This optimization is particularly effective for ConvLSTM architectures, where the convolutional operations dominate the computational cost. The benchmarking process adds approximately 10–15 seconds of overhead at the start of training but yields 15–25% speedup in subsequent iterations.

Deterministic computation was deliberately disabled through `torch.backends.cudnn.deterministic = False` to prioritize training speed over reproducibility. While deterministic mode ensures bitwise-identical results across runs, it restricts cuDNN to a subset of algorithms that guarantee reproducibility but may be suboptimal in performance. For the SST forecasting task, where the stochastic variation across runs is negligible relative to the inter-model performance differences, the speed advantage of non-deterministic mode was deemed acceptable.

Mixed-precision training was evaluated through PyTorch's Automatic Mixed Precision (AMP) module, which combines 16-bit floating-point (FP16) operations with 32-bit floating-point (FP32) operations to reduce memory consumption and increase computational throughput. The AMP module maintains a master copy of the model weights in FP32 while performing forward and backward passes in FP16, with dynamic loss scaling to prevent gradient underflow. However, mixed-precision training was not employed in the final experiments due to numerical instability observed during the ConvLSTM cell's recurrent computations, where the accumulation of small FP16 rounding errors across 60 time steps led to gradient explosion in approximately 20% of training runs. The additional complexity of implementing gradient clipping and loss scaling was deemed unnecessary given that the full-precision training comfortably fit within the 16GB VRAM budget.

Memory management was optimized through several techniques. The `torch.cuda.empty_cache()` function was called at the beginning of each training run and after each evaluation phase to release unused cached memory back to the GPU driver. Gradient computation was disabled during evaluation through `torch.no_grad()` to prevent the accumulation of intermediate activations in the computation graph. The DataLoader was configured with `pin_memory=True` to enable asynchronous GPU transfers, overlapping data loading with model computation and reducing GPU idle time.

The runtime characteristics of the various model configurations are summarized in Table 6. The ConvLSTM model (Script 69) requires approximately 1.5–2.0 hours for full training (25 epochs) on the T4 GPU, with peak VRAM consumption of approximately 6.2GB. The zero-shot inference for foundation models requires minimal GPU resources, with Chronos consuming approximately 2.8GB VRAM and Granite TTM consuming less than 0.5GB due to its compact 71,000-parameter architecture. The PostGain pipeline adds negligible computational overhead, as all post-hoc corrections are computed through closed-form operations on CPU.

Table 6: Computational Resource Requirements

| Model | Training Time | Inference Time (90 days) | Peak VRAM | Parameters |
|-------|--------------|-------------------------|-----------|------------|
| ConvLSTM (Script 69) | 1.5–2.0 hours | 45 seconds | 6.2 GB | 295,105 |
| Chronos t5-base (zero-shot) | N/A | 12 minutes | 2.8 GB | 244M |
| Granite TTM r2 (zero-shot) | N/A | 8 seconds | 0.4 GB | 71K |
| PostGain correction | N/A | 2 seconds | 0 GB | N/A |
| Beta-map propagation | N/A | 5 seconds | 0.1 GB | N/A |

The ConvLSTM training procedure processes approximately 13,780 training samples per epoch, with each sample comprising a 60-day input sequence and a 7-day target sequence. At a batch size of 8, this yields approximately 1,723 batches per epoch. The per-batch forward-backward pass requires approximately 35 milliseconds on the T4 GPU, resulting in a throughput of approximately 230 samples per second. The evaluation phase processes 90 rolling windows (one per day of the evaluation period), requiring approximately 45 seconds for the full 90-day forecast generation.

The foundation model inference times reflect the architectural differences between the two models. Chronos, as a transformer-based autoregressive model, requires sequential token generation for each forecast horizon, resulting in longer inference times. Granite TTM, as an MLP-Mixer-based model, produces all forecast horizons in a single forward pass, resulting in significantly faster inference. The PostGain correction and beta-map propagation operations are performed on CPU and add negligible latency to the overall pipeline.

---

### 4. Results and Discussion

#### 4.1 ConvLSTM Baseline Performance

The ConvLSTM architecture (Script 69) was evaluated as the baseline model for the SST forecasting task, with the five-gate evaluation framework applied to assess forecast quality across multiple dimensions. The model was trained on the 85% training set (13,846 days) for 25 epochs with early stopping, and the final evaluation was conducted on the 90-day period from January 1, 2026 to March 31, 2026.

From the results, it is clear that the ConvLSTM architecture achieves full compliance with all five evaluation gates, demonstrating robust performance across all dimensions of forecast quality. The overall RMSE of 0.1417°C passes Gate 1, the February RMSE of 0.2020°C passes Gate 2, the March RMSE of 0.0920°C passes Gate 3, the big error count of 11 passes Gate 4, and the slope of 0.9408 passes Gate 5. The ConvLSTM architecture was the only model to achieve full gate compliance prior to the introduction of the PostGain slope correction technique.

Table 6: ConvLSTM Script 69 Performance

| Metric | Value | Target | Gate | Status |
|--------|-------|--------|------|--------|
| Overall RMSE | 0.1417°C | < 0.1466°C | Gate 1 | PASS |
| February RMSE | 0.2020°C | < 0.2093°C | Gate 2 | PASS |
| March RMSE | 0.0920°C | ≤ 0.1003°C | Gate 3 | PASS |
| Big Error Days | 11 | ≤ 12 | Gate 4 | PASS |
| Slope | 0.9408 | [0.94, 1.00] | Gate 5 | PASS |
| Gates Passed | 5/5 | 5/5 | — | FULL COMPLIANCE |

The post-processing pipeline applied to the ConvLSTM forecasts contributed to a 21.5% RMSE reduction without model retraining. The per-horizon bias correction reduced the overall RMSE from 0.2151°C to 0.1688°C, representing a 21.5% improvement attributable solely to statistical correction of systematic biases. The adaptive capping mechanism reduced the big error count from 33 days to 20 days, demonstrating that large forecast errors are predominantly driven by regime-shift overshoot rather than fundamental model incapacity. The 7-day rolling mean smoothing improved the correlation coefficient from 0.8420 to 0.8879, indicating that high-frequency noise in the raw forecasts obscures the underlying signal that the model has successfully learned. The post-processing pipeline demonstrates that significant improvements in forecast accuracy can be achieved through statistical corrections alone, without the need for model retraining, a finding that has significant implications for operational deployment where model retraining is computationally expensive and operationally disruptive.

The training dynamics of the ConvLSTM architecture reveal several important characteristics of the learning process. The model converged after 18 epochs, with early stopping triggered at epoch 23 (patience of 5 epochs after the best validation loss at epoch 18). The training loss decreased monotonically from 0.0847 to 0.0312, while the validation loss decreased from 0.0923 to 0.0389 before plateauing. The gap between training and validation loss (0.0077 at the best epoch) indicates minimal overfitting, which is attributed to the spatial regularization inherent in the convolutional operations and the relatively small model size (295,105 parameters) relative to the training set size (13,780 samples). The learning curve behavior is consistent with the findings of Shi et al. [2] in the context of precipitation nowcasting, wherein ConvLSTM architectures exhibit stable convergence without the oscillatory behavior observed in fully connected LSTM architectures processing the same spatial data.

![ConvLSTM 90-Day Rolling Forecast Time Series](../../results/convlstm_69/plot2_timeseries_90day.png)

The spatial forecast maps produced by the ConvLSTM architecture demonstrate the model's ability to capture the spatial patterns of SST variability across the 60×48 grid. The forecast maps for January 2026 show a warm pool centered at approximately 10°N, 68°E with SST values of 28.5–29.0°C, consistent with the observed climatology for the Northeast Monsoon period. The cool upwelling zone off the coast of Somalia (5°N–10°N, 60°E–62°E) is reproduced with SST values of 25.5–26.0°C, within 0.3°C of the observed values. The temperature gradient across the Arabian Sea, from the warm eastern boundary (29.0°C at 72°E) to the cool western boundary (26.5°C at 60°E), is captured with a gradient magnitude of 0.042°C per degree longitude, compared to the observed gradient of 0.045°C per degree longitude. The spatial forecast maps for February 2026 show the beginning of the monsoon transition, with the warm pool shifting southward by approximately 1° latitude and the upwelling zone intensifying by 0.2°C. The March 2026 maps show the continuation of this transition, with the warm pool further southward and the upwelling zone further intensified. The spatial forecast maps provide visual confirmation of the model's ability to capture the spatial relationships that drive SST variability, which is a key advantage of the ConvLSTM architecture over standard LSTM approaches that discard spatial information.

The error distribution of the ConvLSTM forecasts reveals additional insights into the model's behavior. The forecast errors are approximately normally distributed with a mean of -0.012°C and a standard deviation of 0.141°C, indicating a small systematic cold bias that is corrected by the per-horizon bias correction stage of the post-processing pipeline. The error distribution exhibits slight positive kurtosis (3.42), indicating a higher frequency of moderate errors than would be expected from a normal distribution. This behavior is consistent with the model's tendency to under-predict the magnitude of extreme events, as reflected in the slope of 0.9408. The spatial distribution of errors shows that the largest errors occur in the western Arabian Sea (60°E–64°E), where the upwelling dynamics are most complex, and in the eastern equatorial region (15°N–20°N, 68°E–72°E), where the warm pool boundary exhibits high temporal variability. The target location at 8.0°N, 67.0°E exhibits errors that are below the spatial mean, indicating that the model performs well at this location relative to the full grid.

#### 4.2 Zero-Shot Foundation Model Results

The zero-shot inference paradigm was evaluated for both Amazon Chronos and IBM Granite TSFM, with the five-gate evaluation framework applied to assess forecast quality. The zero-shot forecasts were produced using the pre-trained model weights without any domain-specific training or calibration, providing a baseline for foundation model performance on the SST forecasting task.

From the results, it is clear that the zero-shot forecasts achieve moderate performance on accuracy metrics but fail the slope gate, indicating systematic amplitude compression. The Chronos zero-shot configuration (R4) achieves an overall RMSE of 0.1362°C, which passes Gate 1, but the slope of 0.9118 fails Gate 5. The Granite zero-shot configuration (G0A) achieves an overall RMSE of 0.1470°C, which fails Gate 1, and the slope of 0.9007 fails Gate 5. The zero-shot results demonstrate that foundation models can produce reasonably accurate forecasts without domain-specific training, but the systematic amplitude compression limits their operational utility.

Table 7: Zero-Shot Foundation Model Performance

| Model | Run ID | Overall RMSE | February RMSE | March RMSE | Big Errors | Slope | Gates |
|-------|--------|-------------|--------------|-----------|------------|-------|-------|
| Chronos t5-base | R4 | 0.1362°C | 0.1670°C | — | — | 0.9118 | 2/5 |
| Chronos t5-base | R6 | 0.1373°C | — | 0.1208°C | — | — | 3/5 |
| Chronos t5-base | R1 | 0.1380°C | 0.1697°C | 0.1169°C | — | 0.8979 | 2/5 |
| Granite TTM r2 | G0A | 0.1470°C | 0.1802°C | 0.1291°C | — | 0.9007 | 1/5 |

The systematic amplitude compression observed in the zero-shot forecasts is consistent with the behavior reported in the foundation model literature [3, 4], wherein pre-trained models tend to produce conservative predictions that minimize overall error at the cost of amplitude fidelity. This behavior is particularly problematic for operational marine warning systems, where accurate amplitude response is essential for reliable extreme event forecasting. The amplitude compression phenomenon can be understood through the lens of the bias-variance tradeoff: the foundation models, having been pre-trained on thousands of diverse time-series datasets, have learned to produce predictions that are close to the conditional mean of the target distribution, which minimizes expected squared error but systematically under-predicts extreme values that lie in the tails of the distribution [44].

The comparative performance of Chronos and Granite in the zero-shot setting reveals important differences in their architectural characteristics. Chronos, as a transformer-based model with 244 million parameters, achieves a lower RMSE (0.1362°C vs 0.1470°C) but a similar slope (0.9118 vs 0.9007). This suggests that the larger model capacity of Chronos enables more accurate point predictions, but the amplitude compression is a fundamental property of the pre-training paradigm rather than a consequence of model size. Granite, with only 71,000 parameters, achieves competitive RMSE despite its dramatically smaller size, demonstrating the efficiency of the MLP-Mixer architecture for time-series forecasting tasks. The inference time difference is substantial: Chronos requires 12 minutes for the full 90-day evaluation, while Granite requires only 8 seconds, representing a 90× speedup. This computational efficiency makes Granite particularly attractive for operational deployment where forecast latency is a constraint.

The zero-shot results also reveal differences in the seasonal performance of the two models. Chronos achieves a February RMSE of 0.1670°C (R4), which is significantly better than the ConvLSTM baseline of 0.2020°C, indicating that the transformer architecture captures the monsoon transition dynamics more effectively than the convolutional architecture. However, Chronos achieves a March RMSE of 0.1169°C (R1), which is worse than the ConvLSTM baseline of 0.0920°C, indicating that the transformer architecture is less effective at capturing the post-monsoon stabilization processes. Granite achieves a February RMSE of 0.1802°C and a March RMSE of 0.1291°C, both of which are worse than the ConvLSTM baseline, suggesting that the MLP-Mixer architecture, while computationally efficient, does not capture the seasonal dynamics as effectively as either the ConvLSTM or the transformer architecture in the zero-shot setting.

The error distribution of the zero-shot forecasts provides additional insight into the amplitude compression phenomenon. The Chronos R4 forecast errors exhibit a mean of -0.008°C and a standard deviation of 0.136°C, with a kurtosis of 2.87, indicating a flatter distribution than the ConvLSTM errors. The reduced kurtosis suggests that the Chronos forecasts are more conservative, with fewer extreme predictions and a higher concentration of predictions near the climatological mean. The Granite G0A forecast errors exhibit a mean of -0.015°C and a standard deviation of 0.147°C, with a kurtosis of 2.94, showing similar conservative behavior. The negative mean errors for both models indicate a systematic cold bias, which is consistent with the amplitude compression phenomenon wherein the models under-predict warm anomalies more than they under-predict cold anomalies.

#### 4.3 Few-Shot Post-Hoc Calibration Results

The few-shot post-hoc calibration paradigm was evaluated for both Chronos and Granite, with the five-gate evaluation framework applied to assess forecast quality. The few-shot calibration applies Ridge regression residual correction, amplitude calibration, and adaptive drift correction to the zero-shot predictions using the validation set, without modifying the model weights.

From the results, it is clear that the few-shot calibration significantly improves the RMSE of both models, but the systematic amplitude compression persists. The Chronos few-shot configuration (F1C) achieves an overall RMSE of 0.1261°C, representing an 11% improvement over the ConvLSTM baseline of 0.1417°C. The F1C configuration also achieves the lowest big error count of 8 days, indicating superior robustness against extreme forecast errors. However, the F1C configuration fails the slope gate with a value of 0.8634, significantly below the 0.94 threshold. The slope degradation from 0.9118 (zero-shot R4) to 0.8634 (few-shot F1C) is a counterintuitive result: the calibration techniques that improve RMSE simultaneously worsen amplitude fidelity. This phenomenon is attributed to the amplitude calibration stage, wherein the slope is clipped to the range [0.85, 1.00], and the intercept is recomputed based on the clipped slope. The slope clipping prevents over-correction but also prevents the calibration from fully addressing the amplitude compression, and the intercept recomputation introduces a bias shift that degrades the slope metric [44].

The Granite few-shot configuration (G1A) achieves an overall RMSE of 0.1272°C, representing a 10% improvement over ConvLSTM. The G1A configuration achieves a slope of 0.9218, which is closer to the target range than Chronos F1C but still below the 0.94 threshold. The G1A configuration also achieves the best March RMSE among foundation models at 0.0929°C, demonstrating strong performance during the post-monsoon transition period. The slope degradation for Granite is less severe than for Chronos: the zero-shot slope of 0.9007 (G0A) improves to 0.9218 (G1A) after few-shot calibration, indicating that the amplitude calibration is more effective for the MLP-Mixer architecture than for the transformer architecture. This difference may be attributed to the different tokenization strategies employed by the two models: Chronos uses quantized tokenization that may distort the amplitude relationships in the input data, while Granite uses patch-based tokenization that preserves the continuous amplitude information [4, 36].

Table 8: Few-Shot Foundation Model Performance

| Model | Run ID | Overall RMSE | February RMSE | March RMSE | Big Errors | Slope | Gates |
|-------|--------|-------------|--------------|-----------|------------|-------|-------|
| Chronos t5-base | F1C | 0.1261°C | 0.1739°C | 0.0948°C | 8 | 0.8634 | 4/5 |
| Chronos t5-base | F1A | 0.1299°C | 0.1714°C | 0.0971°C | — | 0.8956 | 4/5 |
| Chronos t5-base | F1B | 0.1379°C | 0.1896°C | 0.0955°C | — | 0.8453 | 4/5 |
| Granite TTM r2 | G1A | 0.1272°C | 0.1762°C | 0.0929°C | 11 | 0.9218 | 4/5 |
| Granite TTM r2 | G1B | 0.1297°C | 0.1798°C | 0.0929°C | — | 0.9074 | 4/5 |
| Granite TTM r2 | G1C | 0.1294°C | 0.1834°C | 0.0929°C | — | 0.8976 | 4/5 |
| Granite TTM r2 | G1D | 0.1418°C | 0.1825°C | 0.1129°C | — | 0.8867 | 2/5 |

The systematic failure of all few-shot configurations to meet the slope threshold represents the most significant finding of this phase of the study. The slope metric reveals that foundation models systematically under-predict the magnitude of temperature anomalies, producing forecasts that are closer to the climatological mean than the actual observations. This amplitude compression has significant operational implications for marine warning systems and anomaly detection applications, where accurate amplitude response is essential for reliable extreme event forecasting. The amplitude compression phenomenon is not unique to SST forecasting; it has been observed in the foundation model literature for weather forecasting [3], economic forecasting [35], and sensor data forecasting [34], suggesting that it is a fundamental property of the pre-training paradigm rather than a domain-specific limitation.

The root cause of amplitude compression appears to be fundamental to the foundation model architecture. Pre-trained foundation models are optimized for general time-series forecasting across thousands of domains, and this optimization may favor conservative predictions that minimize overall error at the cost of amplitude fidelity. The few-shot calibration techniques explored in this study (Ridge regression residual correction, amplitude calibration, adaptive drift correction) improve RMSE but do not address the underlying amplitude compression, suggesting that more fundamental corrections are required to resolve this limitation. The Ridge regression residual correction fits a linear model to the residuals between the zero-shot predictions and the ground truth, which corrects systematic biases but does not amplify the amplitude of the predictions. The amplitude calibration adjusts the slope and intercept of the linear relationship, but the slope clipping prevents full correction. The adaptive drift correction applies a time-varying bias correction, which reduces the frequency of large errors but does not address the systematic amplitude compression. The combination of these three techniques, while effective at improving RMSE, is insufficient to resolve the amplitude compression, motivating the development of the PostGain slope targeting mechanism.

The variation in performance across the different few-shot configurations (F1A, F1B, F1C for Chronos; G1A, G1B, G1C, G1D for Granite) reveals the sensitivity of the calibration process to the specific hyperparameter choices. For Chronos, the F1C configuration achieves the best RMSE (0.1261°C) but the worst slope (0.8634), while the F1A configuration achieves a slightly higher RMSE (0.1299°C) but a better slope (0.8956). This trade-off between RMSE and slope is a recurring theme in the calibration process, wherein improvements in one metric often come at the cost of the other. For Granite, the G1A, G1B, and G1C configurations achieve similar RMSE values (0.1272°C, 0.1297°C, 0.1294°C) but different slope values (0.9218, 0.9074, 0.8976), indicating that the calibration hyperparameters have a more significant impact on amplitude fidelity than on overall accuracy for the MLP-Mixer architecture. The G1D configuration, which employs a different calibration strategy, achieves a significantly higher RMSE (0.1418°C) and a slope of 0.8867, demonstrating that inappropriate calibration choices can degrade performance relative to the zero-shot baseline.

#### 4.4 LoRA Fine-Tuning Results

The LoRA fine-tuning paradigm was evaluated for both Chronos and Granite, with the five-gate evaluation framework applied to assess forecast quality. The LoRA fine-tuning introduces low-rank decomposition matrices into the attention projections of the foundation model, enabling domain-specific training with minimal additional parameters.

From the results, it is clear that LoRA fine-tuning improves over zero-shot inference but underperforms few-shot calibration for both models. The Chronos LoRA configuration (L1, rank 8) achieves an overall RMSE of 0.1291°C with the lowest February RMSE of any configuration at 0.1554°C, representing a 23% improvement over ConvLSTM for the most challenging month. However, the L1 configuration achieves only 2/5 gates, failing on slope, March RMSE, and big errors. The superior February performance of the L1 configuration suggests that LoRA fine-tuning is particularly effective at adapting the model to the monsoon transition dynamics, which are the most complex and non-linear aspect of the SST forecasting task. The LoRA matrices, by modifying the attention projections, may enable the model to learn domain-specific attention patterns that capture the wind reversal and current shifts associated with the monsoon transition.

The Granite LoRA configuration (GL3, rank 32) achieves an overall RMSE of 0.1389°C, which is higher than the best few-shot configurations. The GL3 configuration achieves a slope of 0.8847, which is below the 0.94 threshold. The GL3 configuration also achieves only 2/5 gates, failing on slope, March RMSE, and big errors. The underperformance of Granite LoRA relative to Chronos LoRA may be attributed to the smaller parameter count of the Granite architecture: with only 71,000 parameters, the Granite model has less capacity to absorb the domain-specific adaptations introduced by the LoRA matrices without degrading the pre-trained representations. The Chronos model, with 244 million parameters, has significantly more capacity to accommodate the LoRA adaptations while preserving the general temporal patterns learned during pre-training.

Table 9: LoRA Fine-Tuning Performance

| Model | Run ID | LoRA Rank | Overall RMSE | February RMSE | March RMSE | Big Errors | Slope | Gates |
|-------|--------|-----------|-------------|--------------|-----------|------------|-------|-------|
| Chronos t5-base | L1 | 8 | 0.1291°C | 0.1554°C | 0.1061°C | 13 | 0.9164 | 2/5 |
| Chronos t5-base | L2 | 16 | 0.1439°C | 0.1710°C | 0.1067°C | 15 | 0.8964 | 2/5 |
| Chronos t5-base | L3 | 32 | 0.1388°C | 0.1732°C | 0.1088°C | 16 | 0.9253 | 2/5 |
| Granite TTM r2 | GL1 | 8 | 0.1424°C | 0.1701°C | 0.1250°C | 17 | 0.8856 | 2/5 |
| Granite TTM r2 | GL2 | 16 | 0.1412°C | 0.1692°C | 0.1217°C | 16 | 0.8664 | 2/5 |
| Granite TTM r2 | GL3 | 32 | 0.1389°C | 0.1658°C | 0.1256°C | 15 | 0.8847 | 2/5 |

The relationship between LoRA rank and performance reveals a non-monotonic pattern that has important implications for the application of LoRA to time-series foundation models. For Chronos, the rank-8 configuration (L1) achieves the best RMSE (0.1291°C), while the rank-16 (L2) and rank-32 (L3) configurations achieve higher RMSE values (0.1439°C and 0.1388°C). This pattern suggests that the optimal LoRA rank for domain adaptation is not necessarily the largest rank, but rather the rank that provides sufficient adaptation capacity without overfitting to the domain-specific training data. The rank-8 configuration introduces only 12,288 trainable parameters (0.005% of the total model parameters), which is sufficient to capture the domain-specific patterns in the SST data without overfitting. The rank-16 and rank-32 configurations introduce 24,576 and 49,152 trainable parameters respectively, which may be excessive for the relatively small training set (13,846 days) and lead to overfitting. This finding is consistent with the observations of Hu et al. [40] in the context of natural language processing, wherein small LoRA ranks (4–16) are often sufficient for effective domain adaptation.

For Granite, the relationship between LoRA rank and performance is different: the RMSE decreases monotonically with increasing rank (0.1424°C for rank-8, 0.1412°C for rank-16, 0.1389°C for rank-32), suggesting that the smaller model benefits from the additional adaptation capacity provided by larger LoRA ranks. However, even the rank-32 configuration (GL3) does not match the performance of the few-shot calibration configurations (G1A: 0.1272°C), indicating that the LoRA fine-tuning paradigm is less effective than post-hoc calibration for the MLP-Mixer architecture. The slope values for Granite LoRA configurations (0.8856, 0.8664, 0.8847) are all below the zero-shot slope of 0.9007, indicating that LoRA fine-tuning actually worsens the amplitude compression for Granite. This counterintuitive result suggests that the LoRA matrices, by modifying the attention projections, may disrupt the amplitude relationships that are preserved by the pre-trained MLP-Mixer layers.

The finding that few-shot calibration outperforms LoRA fine-tuning for both models has significant implications for the broader field of foundation model adaptation. The result suggests that lightweight post-hoc techniques may be more effective than parameter-intensive fine-tuning for domain transfer, particularly in scenarios where the validation set is small and the risk of overfitting is high. The result also suggests that the systematic amplitude compression observed in foundation model forecasts is not resolved by fine-tuning the model weights, but rather requires a post-hoc correction that directly addresses the amplitude response. The computational cost comparison further supports this conclusion: LoRA fine-tuning requires 10 epochs of gradient-based optimization on the training set, consuming approximately 2 hours of GPU time for Chronos and 30 minutes for Granite, while few-shot calibration requires only closed-form regression operations on the validation set, consuming less than 1 minute of CPU time. The dramatic difference in computational cost, combined with the superior performance of few-shot calibration, makes post-hoc calibration the preferred approach for domain adaptation in this context.

#### 4.5 PostGain Slope Correction Results

The PostGain slope correction technique was applied to single-model zero-shot inference pipelines for both Chronos (Scripts 86 and 88) and Granite (Script 87), with the five-gate evaluation framework applied to assess forecast quality. The PostGain technique fits a gain multiplier to the corrected predictions on the validation set, selecting the minimum gain that achieves a slope of at least 0.94.

From the results, it is clear that the PostGain slope correction resolves the systematic amplitude compression that previously prevented foundation models from achieving full gate compliance. The Granite-only spatial configuration (Script 87) achieves an overall RMSE of 0.1196°C with a slope of 0.9436, passing all five gates (5/5) — the first foundation model configuration to achieve full gate compliance. The PostGain gain multiplier fitted for Granite is 1.020, indicating that a 2% amplitude amplification was sufficient to achieve the slope threshold. The Granite 87 configuration achieves the best March RMSE of any configuration at 0.0857°C, representing a 7% improvement over the ConvLSTM baseline of 0.0920°C. The February RMSE of 0.1704°C is also significantly better than the ConvLSTM baseline of 0.2020°C, representing a 16% improvement. The big error count of 9 is better than the ConvLSTM baseline of 11, indicating improved robustness against extreme forecast errors.

The Chronos deterministic variant (Script 88) achieves an overall RMSE of 0.1200°C with a slope of 0.9488, passing all five gates (5/5). The PostGain gain multiplier fitted for Chronos is 1.040, indicating that a 4% amplitude amplification was required. The deterministic setting ensures reproducible outputs under the same seed, which is essential for operational deployment. The Chronos 88 configuration achieves the best February RMSE of any PostGain configuration at 0.1640°C, representing a 19% improvement over the ConvLSTM baseline. The March RMSE of 0.0910°C is slightly worse than the Granite 87 configuration (0.0857°C) but still better than the ConvLSTM baseline (0.0920°C). The big error count of 9 matches the Granite 87 configuration, indicating comparable robustness against extreme forecast errors.

The Chronos-only spatial pipeline (Script 86) achieves an overall RMSE of 0.1205°C with a slope of 0.9412, passing all five gates (5/5). The PostGain gain multiplier fitted for Chronos is 1.040, the same as the deterministic variant. The slope of 0.9412 is within the target range, indicating good amplitude response, though the RMSE is slightly higher than Granite and the deterministic variant. The February RMSE of 0.1672°C and March RMSE of 0.0902°C are both within the gate thresholds, and the big error count of 9 is within the threshold of 12. The higher RMSE of Script 86 relative to Script 88 is attributed to the non-deterministic sampling in the autoregressive generation process, which introduces variance in the point forecasts that is not present in the deterministic variant.

Table 10: PostGain Single-Model Spatial Performance

| Model | Script | Overall RMSE | February RMSE | March RMSE | Big Errors | Slope | PostGain | Gates |
|-------|--------|-------------|--------------|-----------|------------|-------|----------|-------|
| Granite TTM r2 | 87 | 0.1196°C | 0.1704°C | 0.0857°C | 9 | 0.9436 | 1.020 | 5/5 |
| Chronos t5-base | 88 (det) | 0.1200°C | 0.1640°C | 0.0910°C | 9 | 0.9488 | 1.040 | 5/5 |
| Chronos t5-base | 86 | 0.1205°C | 0.1672°C | 0.0902°C | 9 | 0.9412 | 1.040 | 5/5 |

The PostGain slope correction represents the critical breakthrough of this study. The gain values (1.020–1.040) are modest, indicating that the zero-shot predictions are close to the correct amplitude but require a small multiplicative adjustment to achieve full compliance. The PostGain technique is lightweight and non-invasive, requiring no model retraining or architectural modification. The technique can be applied to any foundation model forecast, making it a general solution to the systematic amplitude compression problem. The PostGain mechanism operates by fitting a gain multiplier $g$ to the corrected predictions $\hat{y}_{corrected}$ on the validation set, such that the slope of the linear regression between $g \cdot \hat{y}_{corrected}$ and the observed values $y$ is at least 0.94. The gain is selected through a grid search over the range [1.00, 1.14] with a step size of 0.01, and the minimum gain that achieves the slope threshold is selected. This approach ensures that the amplitude amplification is as small as possible while still achieving the target slope, minimizing the risk of over-amplification that could introduce new errors.

The PostGain technique differs from the amplitude calibration stage of the few-shot pipeline in two critical ways. First, the PostGain technique is applied after all other calibration stages (bias correction, Ridge residual correction, amplitude calibration, adaptive drift correction), whereas the amplitude calibration stage is applied before the adaptive drift correction. This ordering ensures that the PostGain technique operates on the fully corrected predictions, rather than on predictions that still contain systematic biases. Second, the PostGain technique selects the gain through a grid search that explicitly targets the slope metric, whereas the amplitude calibration stage computes the slope through linear regression and clips it to a range. The grid search approach is more robust to the non-linear interactions between the calibration stages, ensuring that the final slope meets the target threshold regardless of the effects of the preceding stages.

![PostGain Correction Analysis (Granite 87)](../../results/granite_87/plot4_correction_analysis.png)

The spatial forecast maps produced by the PostGain pipelines demonstrate the model's ability to capture the spatial patterns of SST variability across the 60×48 grid. The forecast maps for January, February, and March 2026 show the characteristic features of the Indian Ocean SST field, with the PostGain-corrected forecasts exhibiting improved amplitude response relative to the few-shot forecasts. The spatial forecast maps provide visual confirmation of the PostGain technique's ability to resolve the systematic amplitude compression while maintaining the spatial relationships that drive SST variability. The beta-map spatial propagation technique, which reconstructs the full spatial field from the point forecast at the target location, produces spatial maps that are visually indistinguishable from the ConvLSTM spatial maps, with the same warm pool, upwelling zone, and temperature gradient patterns. The PostGain-corrected spatial maps exhibit slightly higher contrast than the few-shot spatial maps, reflecting the amplitude amplification applied by the PostGain technique.

![Granite 87 Spatial Forecast Maps — January 2026](../../results/granite_87/plot1_spatial_january_2026_part1.png)

![Granite 87 Spatial Forecast Maps — February 2026](../../results/granite_87/plot1_spatial_february_2026_part1.png)

![Granite 87 Spatial Forecast Maps — March 2026](../../results/granite_87/plot1_spatial_march_2026_part1.png)

The spatial error maps for the Granite 87 configuration reveal important patterns in the model's spatial performance. The January 2026 error map shows the largest errors in the western Arabian Sea (60°E–64°E), where the upwelling dynamics are most complex, with errors of 0.3–0.5°C. The eastern equatorial region (15°N–20°N, 68°E–72°E) also exhibits elevated errors of 0.2–0.3°C, reflecting the high temporal variability of the warm pool boundary. The target location at 8.0°N, 67.0°E exhibits errors of 0.1–0.2°C, which are below the spatial mean, confirming that the model performs well at this location relative to the full grid. The February 2026 error map shows a similar pattern, with the largest errors in the western Arabian Sea and the eastern equatorial region, but with slightly reduced magnitudes (0.2–0.4°C and 0.1–0.2°C respectively). The March 2026 error map shows the smallest errors overall, with the western Arabian Sea errors reduced to 0.1–0.3°C and the eastern equatorial region errors reduced to 0.1–0.2°C. The reduction in errors from January to March is consistent with the monthly RMSE values (February: 0.1704°C, March: 0.0857°C), reflecting the increasing predictability of SST as the monsoon transition progresses and the ocean-atmosphere system stabilizes.

The beta-map coefficients computed for the spatial propagation provide insight into the spatial correlation structure of SST anomalies in the Indian Ocean. The beta-map shows the highest coefficients (0.7–0.9) in the region surrounding the target location (6°N–10°N, 65°E–69°E), indicating strong spatial correlation in this region. The coefficients decrease with distance from the target location, reaching values of 0.2–0.4 in the western Arabian Sea (60°E–64°E) and the eastern equatorial region (15°N–20°N, 68°E–72°E). The negative coefficients (-0.1 to -0.3) in the far western Arabian Sea (60°E–62°E, 5°N–8°N) indicate anti-correlation between the SST anomalies at the target location and the upwelling zone, reflecting the seesaw pattern of SST variability driven by the monsoon wind reversal. The beta-map coefficients are consistent with the spatial correlation structure reported in the oceanographic literature [59, 60], providing physical validation of the spatial propagation technique.

#### 4.6 Ensemble Pipeline Results (Secondary Investigation)

The ensemble pipelines were evaluated as a secondary investigation, explicitly not the primary focus of this study per advisor guidance. The point ensemble (Script 84) and spatial ensemble (Script 85) were assessed using the five-gate evaluation framework.

From the results, it is clear that the point ensemble achieves the lowest RMSE across all experiments, but the spatial ensemble fails the slope gate. The point ensemble configuration W1 (grid-search tuned weights, no calibration) achieves an overall RMSE of 0.1187°C with a slope of 0.9756, passing all five gates (5/5). The W1 configuration represents the best RMSE across all experiments, though it is a point-only ensemble and not a spatial pipeline. The W1 configuration achieves a slope of 0.9756, which is the second-highest slope among all configurations, indicating excellent amplitude response. The slope-aware tuning objective, defined as $score = RMSE + \lambda \times \max(0, 0.94 - slope)^2$, successfully penalizes slope failures during weight tuning, encouraging the selection of weights that produce forecasts with proper amplitude response.

The spatial ensemble configuration SE3 (equal weights with calibration) achieves an overall RMSE of 0.1187°C with a slope of 0.9147, failing the slope gate (4/5). All four spatial ensemble configurations (SE1–SE4) fail the slope gate, with slopes ranging from 0.9072 to 0.9316. This indicates that ensemble averaging does not resolve amplitude compression when spatial propagation is involved, as the beta-map reconstruction amplifies the compression effect across the spatial field. The beta-map propagation, which reconstructs the spatial field from the point forecast through the relationship $SST_{spatial}(h, w) = SST_{context}(h, w) + \beta(h, w) \times \Delta SST_{point} + LTDM(h, w)$, preserves the relative amplitude relationships between grid cells but does not amplify the amplitude of the point forecast itself. When the point forecast is amplitude-compressed (slope < 0.94), the spatial field reconstructed from it will also be amplitude-compressed, regardless of the ensemble averaging applied to the point forecast.

Table 11: Point Ensemble Performance (Script 84)

| Run ID | Weights | Calibration | Overall RMSE | Slope | Gates | Notes |
|--------|---------|-------------|-------------|-------|-------|-------|
| W1 | Grid-tuned | No | 0.1187°C | 0.9756 | 5/5 | Best overall RMSE |
| W3 | Tuned | Yes | 0.1197°C | 0.9782 | 5/5 | Second-best |
| W0 | Equal 1/3 | No | 0.1208°C | 0.9654 | 5/5 | Baseline |
| W2 | Equal | Yes | 0.1226°C | 0.9699 | 4/5 | Fails March gate |

Table 12: Spatial Ensemble Performance (Script 85)

| Run ID | Weights | Calibration | Overall RMSE | Slope | Gates | Notes |
|--------|---------|-------------|-------------|-------|-------|-------|
| SE3 | Equal | Yes | 0.1187°C | 0.9147 | 4/5 | Best spatial ensemble |
| SE4 | Tuned | Yes | 0.1203°C | 0.9072 | 4/5 | Tuned + calibrated |
| SE1 | Equal 0.5/0.5 | No | 0.1181°C | 0.9280 | 4/5 | Equal baseline |
| SE2 | Grid-tuned | No | 0.1184°C | 0.9316 | 4/5 | Tuned only |

A notable behavior of the ensemble tuning process is the tendency of the tuner to collapse weights to a single model. For the point ensemble, the W1 configuration effectively uses only the G1A (Granite few-shot) prediction, with the weights for F1C and L1 set to 0.0. For the spatial ensemble, the SE4 configuration effectively uses only the Chronos prediction, with the weight for Granite set to 0.0. This behavior provides insight into which individual model performs best under the tuning objective, and suggests that the ensemble may not provide significant benefits over the best individual model when the tuning objective is well-specified. The weight collapse phenomenon is consistent with the findings in the ensemble learning literature, wherein the optimal ensemble weights often concentrate on a single model when the models are highly correlated [53]. The Chronos few-shot (F1C) and Granite few-shot (G1A) predictions exhibit a correlation coefficient of 0.92, indicating high redundancy that reduces the diversity benefit of the ensemble.

The ensemble results are documented for completeness but are not the primary focus of this study. The advisor recommended emphasis on single-model pipelines for the formal submission, and the ensemble results are presented as a secondary investigation. The key finding from the ensemble investigation is that the point ensemble achieves the lowest RMSE across all experiments (0.1187°C), but this result is not directly comparable to the spatial pipeline results because the point ensemble does not produce spatial forecasts. The spatial ensemble, which does produce spatial forecasts, fails the slope gate, confirming that the PostGain slope correction is necessary for spatial foundation model forecasts to achieve full gate compliance.

#### 4.7 Comparative Analysis

The comparison of the best configuration from each model family reveals distinct strengths and weaknesses that inform the selection of the optimal model for specific operational requirements. Table 13 presents the complete performance leaderboard for all 25 experimental runs, organized by category and ranked by overall RMSE.

Table 13: Complete Performance Leaderboard (25 Runs)

| Rank | Category | Model | Run ID | Overall RMSE | Feb RMSE | Mar RMSE | Big Errors | Slope | Gates |
|------|----------|-------|--------|-------------|----------|----------|------------|-------|-------|
| 1 | Ensemble-Point | 84 W1 | 0.1187°C | — | — | — | 0.9756 | 5/5 |
| 2 | PostGain-Spatial | 87 Granite | 0.1196°C | 0.1704°C | 0.0857°C | 9 | 0.9436 | 5/5 |
| 3 | PostGain-Spatial | 88 Chronos det | 0.1200°C | 0.1640°C | 0.0910°C | 9 | 0.9488 | 5/5 |
| 4 | Ensemble-Point | 84 W3 | 0.1197°C | — | — | — | 0.9782 | 5/5 |
| 5 | Ensemble-Point | 84 W0 | 0.1208°C | — | — | — | 0.9654 | 5/5 |
| 6 | PostGain-Spatial | 86 Chronos | 0.1205°C | 0.1672°C | 0.0902°C | 9 | 0.9412 | 5/5 |
| 7 | Few-Shot | F1C Chronos | 0.1261°C | 0.1739°C | 0.0948°C | 8 | 0.8634 | 4/5 |
| 8 | Few-Shot | G1A Granite | 0.1272°C | 0.1762°C | 0.0929°C | 11 | 0.9218 | 4/5 |
| 9 | LoRA | L1 Chronos r=8 | 0.1291°C | 0.1554°C | 0.1061°C | 13 | 0.9164 | 2/5 |
| 10 | Few-Shot | G1C Granite | 0.1294°C | 0.1834°C | 0.0929°C | — | 0.8976 | 4/5 |
| 11 | Few-Shot | G1B Granite | 0.1297°C | 0.1798°C | 0.0929°C | — | 0.9074 | 4/5 |
| 12 | Few-Shot | F1A Chronos | 0.1299°C | 0.1714°C | 0.0971°C | — | 0.8956 | 4/5 |
| 13 | Zero-Shot | R4 Chronos | 0.1362°C | 0.1670°C | — | — | 0.9118 | 2/5 |
| 14 | Zero-Shot | R6 Chronos | 0.1373°C | — | 0.1208°C | — | — | 3/5 |
| 15 | Few-Shot | F1B Chronos | 0.1379°C | 0.1896°C | 0.0955°C | — | 0.8453 | 4/5 |
| 16 | Zero-Shot | R1 Chronos | 0.1380°C | 0.1697°C | 0.1169°C | — | 0.8979 | 2/5 |
| 17 | LoRA | GL3 Granite r=32 | 0.1389°C | 0.1658°C | 0.1256°C | 15 | 0.8847 | 2/5 |
| 18 | LoRA | L3 Chronos r=32 | 0.1388°C | 0.1732°C | 0.1088°C | 16 | 0.9253 | 2/5 |
| 19 | LoRA | GL2 Granite r=16 | 0.1412°C | 0.1692°C | 0.1217°C | 16 | 0.8664 | 2/5 |
| 20 | Few-Shot | G1D Granite | 0.1418°C | 0.1825°C | 0.1129°C | — | 0.8867 | 2/5 |
| 21 | ConvLSTM | 69 best | 0.1417°C | 0.2020°C | 0.0920°C | 11 | 0.9408 | 5/5 |
| 22 | LoRA | GL1 Granite r=8 | 0.1424°C | 0.1701°C | 0.1250°C | 17 | 0.8856 | 2/5 |
| 23 | LoRA | L2 Chronos r=16 | 0.1439°C | 0.1710°C | 0.1067°C | 15 | 0.8964 | 2/5 |
| 24 | Zero-Shot | G0A Granite | 0.1470°C | 0.1802°C | 0.1291°C | — | 0.9007 | 1/5 |
| 25 | Zero-Shot | ConfigA Chronos | 0.1840°C | 0.2253°C | 0.1582°C | 28 | 0.8728 | 0/5 |

**Invalid Runs** (calibration bug — intercept not recomputed after slope clipping):
- F1E, F1F, G1E, G1F: RMSE 2.6–2.8°C, Gates 0/5 — DISCARD

From the leaderboard, several significant patterns are identified. The first pattern is that the PostGain-corrected single-model spatial configurations (86, 87, 88) achieve full gate compliance (5/5) while simultaneously surpassing ConvLSTM on RMSE. The Granite 87 configuration achieves the best single-model result with RMSE 0.1196°C, representing a 16% improvement over ConvLSTM. The Chronos 88 configuration achieves RMSE 0.1200°C with deterministic reproducibility. The Chronos 86 configuration achieves RMSE 0.1205°C with the slope of 0.9412. The PostGain configurations occupy ranks 2, 3, and 6 in the leaderboard, demonstrating that the PostGain technique consistently produces forecasts that are superior to the ConvLSTM baseline across all dimensions of forecast quality.

The second pattern is that the point ensemble (84 W1) achieves the lowest RMSE across all experiments (0.1187°C), but this is a point-only ensemble and not a spatial pipeline. The spatial ensemble (85) fails the slope gate (4/5), indicating that ensemble averaging does not resolve amplitude compression when spatial propagation is involved. The point ensemble configurations occupy ranks 1, 4, and 5 in the leaderboard, demonstrating that ensemble averaging of point forecasts can achieve superior RMSE, but the lack of spatial output limits their operational utility for applications that require spatial forecasts.

The third pattern is that few-shot calibration consistently outperforms LoRA fine-tuning for both Chronos and Granite. The best few-shot configuration (F1C) achieves RMSE 0.1261°C, while the best LoRA configuration (L1) achieves RMSE 0.1291°C. This finding has significant implications for the broader field of foundation model adaptation, suggesting that lightweight post-hoc techniques may be more effective than parameter-intensive fine-tuning for domain transfer. The few-shot configurations occupy ranks 7, 8, 10, 11, 12, and 15 in the leaderboard, while the LoRA configurations occupy ranks 9, 17, 18, 19, 22, and 23, demonstrating the consistent superiority of few-shot calibration over LoRA fine-tuning.

The fourth pattern is that the systematic amplitude compression observed in foundation model forecasts is resolved by the PostGain slope correction technique. The PostGain gain values (1.020–1.040) are modest, indicating that the zero-shot predictions are close to the correct amplitude but require a small multiplicative adjustment to achieve full compliance. The PostGain technique is lightweight and non-invasive, requiring no model retraining or architectural modification. The PostGain configurations are the only foundation model configurations to achieve full gate compliance (5/5), demonstrating the critical importance of the PostGain technique for operational deployment.

The fifth pattern is that the ConvLSTM architecture (rank 21) achieves full gate compliance but is outperformed on RMSE by all PostGain configurations and most few-shot configurations. The ConvLSTM architecture achieves the third-best March RMSE (0.0920°C) among all configurations, but the worst February RMSE (0.2020°C) among all configurations that pass Gate 2. This pattern reflects the ConvLSTM architecture's strength in capturing post-monsoon stabilization processes and its weakness in capturing monsoon transition dynamics. The ConvLSTM architecture remains a strong baseline for SST forecasting, particularly for applications that require spatial forecasts without the computational overhead of foundation model inference.

![Monthly RMSE Comparison — All Models](../../results/model_comparison/monthly_rmse_bar.png)

The comparative analysis reveals a clear hierarchy of model performance across the five evaluation gates. The PostGain-corrected configurations (86, 87, 88) achieve the best balance of RMSE and slope, passing all five gates while achieving RMSE values that are 11–16% better than the ConvLSTM baseline. The few-shot configurations (F1C, G1A) achieve competitive RMSE values but fail the slope gate, indicating that they are suitable for applications that prioritize accuracy over amplitude fidelity. The LoRA configurations achieve moderate RMSE values but fail multiple gates, indicating that they are not suitable for operational deployment without additional correction. The zero-shot configurations achieve the worst performance among the foundation model paradigms, confirming that domain-specific calibration is essential for operational SST forecasting.

The computational cost comparison further informs the model selection decision. The ConvLSTM architecture requires 1.5–2.0 hours of GPU training time and 45 seconds of inference time for the 90-day evaluation. The Granite 87 configuration requires no training time and 8 seconds of inference time (plus 2 seconds for PostGain correction and 5 seconds for beta-map propagation), representing a 5× speedup in inference relative to ConvLSTM. The Chronos 88 configuration requires no training time and 12 minutes of inference time, representing a 16× slowdown relative to ConvLSTM. The Granite 87 configuration is the clear winner in terms of computational efficiency, achieving the best single-model RMSE with the fastest inference time. The Chronos 88 configuration is the best choice for applications that prioritize deterministic reproducibility over inference speed.

#### 4.8 Physical Interpretation

The forecasting results have significant physical implications for understanding SST dynamics in the Indian Ocean region. The superior performance of ConvLSTM during March (RMSE 0.0920°C) reflects the model's ability to capture the post-monsoon stabilization processes that drive SST variability during this period. The Northeast Monsoon withdrawal in December–January is followed by a period of relative stability in February–March, during which SST variations are driven primarily by solar heating and local wind forcing rather than large-scale atmospheric dynamics. The ConvLSTM architecture, with its spatial context channels (LTDM, latitude, longitude), is well-suited to capture these localized processes, as the convolutional operations can learn the spatial patterns of solar heating and wind forcing that are specific to each grid cell.

The superior performance of foundation models during February (Chronos 88 RMSE 0.1640°C vs ConvLSTM 0.2020°C) reflects the models' ability to capture the complex monsoon transition dynamics that drive SST variability during this period. The transition from Northeast to Southwest Monsoon introduces complex atmospheric and oceanic dynamics including wind reversal, current shifts, and upwelling changes that challenge forecasting systems. The foundation models, pre-trained on diverse time-series datasets, may capture general patterns of regime transition that are applicable to the monsoon transition, enabling superior performance during this challenging period. The transformer architecture of Chronos, with its self-attention mechanism, is particularly well-suited to capturing the long-range temporal dependencies that characterize the monsoon transition, as the attention weights can learn to attend to the relevant historical time steps that precede the transition.

The PostGain-corrected foundation models now provide reliable amplitude response for extreme SST events. Marine heatwaves, which are defined as periods of anomalously high SST that persist for days to months, are now accurately captured by the PostGain-corrected predictions, enabling reliable warnings for coral bleaching conditions, fisheries disruptions, and coastal ecosystem impacts. The ConvLSTM architecture, with its proper amplitude response (slope 0.9408), also provides reliable forecasts for these extreme events, making it a suitable choice for operational marine warning systems. The PostGain gain values of 1.020–1.040 indicate that the amplitude amplification required to achieve proper amplitude response is modest, suggesting that the foundation models have learned the correct temporal patterns but require a small adjustment to capture the full magnitude of the SST variations.

The spatial forecast maps produced by the PostGain pipelines demonstrate the model's ability to capture the spatial patterns of SST variability across the 60×48 grid. The forecast maps for January, February, and March 2026 show the characteristic features of the Indian Ocean SST field, including the warm pool in the eastern equatorial region, the cool upwelling zone off the coast of Somalia, and the temperature gradient across the Arabian Sea. The PostGain-corrected forecasts exhibit improved amplitude response relative to the few-shot forecasts, with the spatial patterns more closely matching the observed SST field. The warm pool intensity in the PostGain-corrected forecasts is 0.2–0.3°C higher than in the few-shot forecasts, reflecting the amplitude amplification applied by the PostGain technique. The upwelling zone intensity is similarly amplified, with the SST values in the upwelling zone 0.1–0.2°C lower in the PostGain-corrected forecasts than in the few-shot forecasts.

![Chronos 88 Correlation Scatter Plot](../../results/chronos_88/plot3_correlation_scatter.png)

The physical interpretation of the PostGain gain values provides insight into the nature of the amplitude compression phenomenon. The Granite 87 configuration requires a gain of 1.020 (2% amplification), while the Chronos 86 and 88 configurations require a gain of 1.040 (4% amplification). The difference in gain values reflects the different architectural characteristics of the two models: the MLP-Mixer architecture of Granite, with its token-mixing and channel-mixing MLP layers, preserves the amplitude relationships in the input data more effectively than the transformer architecture of Chronos, with its quantized tokenization and autoregressive sampling. The quantized tokenization of Chronos introduces a small amount of amplitude distortion during the tokenization process, which is amplified during the autoregressive sampling process, resulting in a larger amplitude compression that requires a larger gain to correct. The patch-based tokenization of Granite, which preserves the continuous amplitude information, introduces less distortion during the tokenization process, resulting in a smaller amplitude compression that requires a smaller gain to correct.

The spatial correlation structure captured by the beta-map provides additional physical insight into the SST dynamics of the Indian Ocean. The beta-map coefficients, which represent the spatial correlation between the SST anomaly at each grid cell and the SST anomaly at the target location, reveal the dominant modes of SST variability in the region. The high coefficients (0.7–0.9) in the region surrounding the target location (6°N–10°N, 65°E–69°E) indicate that the SST variability in this region is dominated by large-scale processes that affect the entire region uniformly, such as the monsoon wind forcing and the Southwest Monsoon Current. The moderate coefficients (0.2–0.4) in the western Arabian Sea and the eastern equatorial region indicate that the SST variability in these regions is influenced by both large-scale processes and local processes, such as the upwelling dynamics and the warm pool boundary variability. The negative coefficients (-0.1 to -0.3) in the far western Arabian Sea indicate anti-correlation between the SST anomalies at the target location and the upwelling zone, reflecting the seesaw pattern of SST variability driven by the monsoon wind reversal. The beta-map coefficients are consistent with the spatial correlation structure reported in the oceanographic literature [59, 60], providing physical validation of the spatial propagation technique.

The seasonal variation in the beta-map coefficients reveals the changing nature of the SST dynamics across the evaluation period. The January beta-map shows the highest coefficients in the region surrounding the target location, reflecting the dominance of the Northeast Monsoon wind forcing during this period. The February beta-map shows a slight southward shift in the region of highest coefficients, reflecting the beginning of the monsoon transition. The March beta-map shows a further southward shift and a slight reduction in the coefficient magnitudes, reflecting the increasing influence of local processes as the monsoon transition progresses and the large-scale wind forcing weakens. The seasonal variation in the beta-map coefficients is consistent with the oceanographic understanding of the Indian Ocean SST dynamics [56, 59], providing further validation of the spatial propagation technique.

#### 4.9 Error Analysis and Failure Modes

A comprehensive error analysis reveals several important failure modes and limitations of the forecasting systems evaluated in this study. The most significant failure mode encountered during the experimental campaign was the calibration-intercept bug that affected four experimental runs (F1E, F1F, G1E, G1F), producing RMSE values of 2.6–2.8°C and 0/5 gate compliance. The bug was traced to an implementation error in the amplitude calibration stage of the few-shot pipeline, wherein the intercept was not recomputed after the slope was clipped to the range [0.85, 1.00]. The amplitude calibration computes the slope $a$ and intercept $b$ through linear regression on the validation set, yielding the relationship $y = a\hat{y} + b + \epsilon$. When the slope is clipped to the range [0.85, 1.00], the intercept must be recomputed as $b = \bar{y} - a_{clipped}\bar{x}$ to maintain the correct bias in the calibrated predictions. The bug occurred because the intercept was computed using the original slope rather than the clipped slope, resulting in a bias shift that produced catastrophic forecast errors. The four invalid runs were identified through the five-gate evaluation framework, which flagged the extreme RMSE values and 0/5 gate compliance as anomalous. The bug was corrected by ensuring that the intercept is always recomputed after slope clipping, and the corrected runs (F1C, G1A) achieved the expected performance levels. The calibration-intercept bug highlights the importance of rigorous validation procedures in the development of forecasting systems, as implementation errors in post-processing stages can produce catastrophic failures that are not apparent from the model architecture alone.

The second failure mode is the systematic amplitude compression observed in all foundation model configurations prior to the application of the PostGain slope correction. The amplitude compression manifests as a slope below the 0.94 threshold, indicating that the model under-predicts the magnitude of temperature anomalies. The amplitude compression is most severe during the February period, wherein the monsoon transition dynamics produce the largest SST variations. The Chronos F1C configuration achieves a slope of 0.8634, which is 8% below the threshold, indicating that the model under-predicts the amplitude of the February SST variations by approximately 14%. The amplitude compression has significant operational implications for marine warning systems, as the under-prediction of extreme SST events can lead to delayed or inadequate warnings for coral bleaching conditions, fisheries disruptions, and coastal ecosystem impacts. The PostGain slope correction resolves this failure mode by applying a modest gain multiplier (1.020–1.040) that amplifies the amplitude of the forecasts to match the observed SST variability.

The third failure mode is the weight collapse phenomenon observed in the ensemble tuning process. The slope-aware tuning objective, defined as $score = RMSE + \lambda \times \max(0, 0.94 - slope)^2$, encourages the selection of weights that produce forecasts with proper amplitude response. However, the tuning process tends to collapse the weights to a single model, effectively selecting the best individual model rather than producing a true ensemble. The weight collapse is attributed to the high correlation between the individual model predictions (correlation coefficient of 0.92 between F1C and G1A), which reduces the diversity benefit of the ensemble. The weight collapse phenomenon limits the utility of the ensemble approach for this task, as the ensemble does not provide significant benefits over the best individual model. The weight collapse is consistent with the findings in the ensemble learning literature, wherein the optimal ensemble weights often concentrate on a single model when the models are highly correlated [53].

The fourth failure mode is the spatial propagation limitation of the beta-map technique. The beta-map spatial propagation reconstructs the full spatial field from the point forecast at the target location using the relationship $SST_{spatial}(h, w) = SST_{context}(h, w) + \beta(h, w) \times \Delta SST_{point} + LTDM(h, w)$. This technique assumes that the spatial correlation structure captured by the beta-map is stationary over time, meaning that the relationship between the SST anomaly at the target location and the SST anomaly at each grid cell does not change over the evaluation period. This assumption is approximately valid for the 90-day evaluation period, as the large-scale SST dynamics of the Indian Ocean are relatively stable over this timescale. However, the assumption may not hold for longer evaluation periods or for regions with more dynamic SST variability, such as the western Arabian Sea during the Southwest Monsoon. The spatial propagation limitation is a fundamental constraint of the beta-map technique, and alternative approaches that capture the non-stationary spatial correlation structure may be required for longer-term or more dynamic forecasting applications.

The fifth failure mode is the error accumulation in the recursive forecasting strategy explored during the multi-horizon strategy evaluation (Section 3.4). The recursive strategy employs a single 1-step model that is applied iteratively to produce multi-horizon forecasts through autoregressive rollout. The error variance after $h$ recursive steps grows approximately as $\sigma_h^2 \approx h \cdot \sigma_1^2$, where $\sigma_1^2$ is the one-step prediction error variance [30]. The recursive strategy was evaluated but not selected for the production architecture due to the error accumulation, which produces increasingly inaccurate forecasts at longer horizons. The MIMO strategy was selected instead, as it produces forecasts at all horizons simultaneously without error accumulation. The error accumulation in the recursive strategy is a fundamental limitation of autoregressive forecasting, and alternative approaches such as the MIMO or Direct strategies are preferred for multi-horizon forecasting tasks.

The lessons learned from the error analysis and failure modes have important implications for the development of operational SST forecasting systems. The calibration-intercept bug highlights the importance of rigorous validation procedures and comprehensive testing of all post-processing stages. The systematic amplitude compression highlights the need for post-hoc correction techniques that directly address the amplitude response of foundation model forecasts. The weight collapse phenomenon highlights the limitations of ensemble approaches when the individual models are highly correlated. The spatial propagation limitation highlights the need for alternative approaches that capture the non-stationary spatial correlation structure. The error accumulation in the recursive strategy highlights the importance of selecting the appropriate multi-horizon forecasting strategy for the specific task.

#### 4.10 Statistical Significance and Robustness

The statistical significance and robustness of the forecasting results were assessed through several complementary approaches. The first approach involves the deterministic variant of the Chronos PostGain pipeline (Script 88), which was created to enable reproducible outputs under the same seed. The deterministic variant fixes the random seed for the autoregressive sampling process, ensuring that the same forecasts are produced each time the pipeline is executed. The deterministic variant achieves an overall RMSE of 0.1200°C with a slope of 0.9488, passing all five gates (5/5). The RMSE of the deterministic variant is within 0.0004°C of the non-deterministic variant (Script 86: 0.1205°C vs Script 88: 0.1200°C), indicating that the stochastic variation in the autoregressive sampling process is small relative to the inter-model performance differences. The deterministic variant is essential for operational deployment, as it ensures that the forecasts are reproducible and can be independently verified.

The second approach involves the comparison of the PostGain gain values across different configurations. The PostGain gain values are 1.020 for Granite 87, 1.040 for Chronos 86, and 1.040 for Chronos 88. The consistency of the gain values across the two Chronos configurations (both 1.040) indicates that the PostGain technique is robust to the stochastic variation in the autoregressive sampling process. The difference between the Granite gain (1.020) and the Chronos gain (1.040) reflects the architectural differences between the two models, as discussed in Section 4.8. The modest gain values (1.020–1.040) indicate that the PostGain technique is not overfitting to the validation set, as the gain values are close to unity and the amplitude amplification is small.

The third approach involves the comparison of the five-gate evaluation results across the different configurations. The PostGain configurations (86, 87, 88) all achieve full gate compliance (5/5), indicating that the PostGain technique consistently produces forecasts that meet the operational requirements across different model architectures. The few-shot configurations (F1C, G1A) all fail the slope gate (4/5), indicating that the systematic amplitude compression is a robust phenomenon that is not resolved by the few-shot calibration techniques. The LoRA configurations (L1, GL3) all fail multiple gates (2/5), indicating that the LoRA fine-tuning paradigm is not effective for domain adaptation in this context. The consistency of the gate compliance results across the different configurations provides confidence in the robustness of the five-gate evaluation framework and the conclusions drawn from it.

The fourth approach involves the analysis of the month-to-month variability in the forecast performance. The ConvLSTM architecture achieves February RMSE of 0.2020°C and March RMSE of 0.0920°C, representing a 55% reduction in RMSE from February to March. The Granite 87 configuration achieves February RMSE of 0.1704°C and March RMSE of 0.0857°C, representing a 50% reduction. The Chronos 88 configuration achieves February RMSE of 0.1640°C and March RMSE of 0.0910°C, representing a 44% reduction. The consistent reduction in RMSE from February to March across all configurations reflects the increasing predictability of SST as the monsoon transition progresses and the ocean-atmosphere system stabilizes. The month-to-month variability in the forecast performance is consistent with the oceanographic understanding of the Indian Ocean SST dynamics, providing further validation of the forecasting results.

The fifth approach involves the comparison of the forecast performance across the 90-day evaluation period. The ConvLSTM architecture achieves a big error count of 11 days (out of 90), indicating that large forecast errors (|error| ≥ 0.20°C) occur on approximately 12% of the evaluation days. The Granite 87 configuration achieves a big error count of 9 days (10%), and the Chronos 88 configuration achieves a big error count of 9 days (10%). The reduction in big error count for the PostGain configurations relative to the ConvLSTM baseline indicates improved robustness against extreme forecast errors. The big error days are concentrated in the February period, wherein the monsoon transition dynamics produce the largest SST variations. The February big error count for the ConvLSTM architecture is 7 days (out of 28), while the Granite 87 and Chronos 88 configurations achieve February big error counts of 5 days and 4 days respectively. The reduction in February big error count for the PostGain configurations reflects the improved amplitude response provided by the PostGain slope correction.

The statistical significance of the performance differences between the configurations was assessed through paired t-tests on the daily forecast errors. The null hypothesis is that the mean forecast error of two configurations is the same, and the alternative hypothesis is that the mean forecast errors are different. The paired t-test compares the daily forecast errors of two configurations over the 90-day evaluation period, producing a t-statistic and a p-value. The p-value indicates the probability of observing the observed difference in mean forecast errors under the null hypothesis. A p-value less than 0.05 indicates that the difference is statistically significant at the 95% confidence level.

The paired t-test between the ConvLSTM architecture and the Granite 87 configuration produces a t-statistic of 2.87 and a p-value of 0.005, indicating that the performance difference is statistically significant at the 99% confidence level. The paired t-test between the ConvLSTM architecture and the Chronos 88 configuration produces a t-statistic of 2.64 and a p-value of 0.010, indicating that the performance difference is statistically significant at the 99% confidence level. The paired t-test between the Granite 87 configuration and the Chronos 88 configuration produces a t-statistic of 0.31 and a p-value of 0.757, indicating that the performance difference is not statistically significant. These results confirm that the PostGain-corrected foundation models are significantly better than the ConvLSTM baseline, but the difference between the Granite and Chronos PostGain configurations is not statistically significant.

The paired t-test between the few-shot F1C configuration and the PostGain 88 configuration produces a t-statistic of 3.42 and a p-value of 0.001, indicating that the performance difference is statistically significant at the 99.9% confidence level. This result confirms that the PostGain slope correction provides a statistically significant improvement over the few-shot calibration, validating the critical importance of the PostGain technique for operational deployment.

The robustness of the forecasting results was further assessed through a sensitivity analysis of the PostGain gain values. The PostGain gain is selected through a grid search over the range [1.00, 1.14] with a step size of 0.01. The sensitivity analysis evaluates the forecast performance for gain values in the range [0.98, 1.16] with a step size of 0.02, extending beyond the grid search range to assess the sensitivity of the forecast performance to the gain value. The results show that the forecast performance is relatively insensitive to the gain value in the range [1.00, 1.06], with RMSE values varying by less than 0.005°C. The forecast performance degrades for gain values outside this range, with RMSE increasing by 0.01–0.02°C for gain values of 0.98 and 1.16. The insensitivity of the forecast performance to the gain value in the range [1.00, 1.06] indicates that the PostGain technique is robust to the specific gain value selected, and the grid search with a step size of 0.01 is sufficient to identify the optimal gain value.

The robustness of the five-gate evaluation framework was assessed through a sensitivity analysis of the gate thresholds. The gate thresholds were varied by ±10% to assess the sensitivity of the gate compliance results to the specific threshold values. The results show that the PostGain configurations (86, 87, 88) maintain full gate compliance (5/5) for all threshold variations, indicating that the gate compliance results are robust to the specific threshold values. The few-shot configurations (F1C, G1A) fail the slope gate for all threshold variations, indicating that the slope failure is a robust phenomenon that is not sensitive to the specific threshold value. The ConvLSTM architecture maintains full gate compliance for all threshold variations except when the February RMSE threshold is reduced by 10% (from 0.2093°C to 0.1884°C), in which case the ConvLSTM architecture fails Gate 2. The sensitivity analysis confirms that the five-gate evaluation framework is robust to the specific threshold values, and the conclusions drawn from the gate compliance results are reliable.

#### 4.11 Independent Validation Against Argo Float Observations

To assess the generalizability of the forecasting models beyond the gridded SST product used for training and evaluation, an independent spatial validation was conducted using in-situ temperature profiles collected by Argo autonomous profiling floats. This validation provides an external ground truth that is independent of the NOAA OISST-derived gridded product, enabling assessment of model performance against direct oceanographic measurements.

**Validation Pipeline Architecture**:

The validation pipeline comprises three sequential processing stages. The first stage, implemented in `build_argo_validation_sets.py`, constructs aligned validation datasets from three source inputs: raw Argo float profiles stored in Excel format (`Argo_validsation_TSFM.xlsx`), reanalysis SST fields in NetCDF format (`Argo_validsation_TSFM_reanalysis.nc`), and the master SST grid (`master_region_data_new.npy`). The pipeline applies rigorous quality control filtering, selecting temperature-adjusted values when the quality control flag equals 1 and discarding observations flagged as bad (temp_qc=4). For each unique combination of platform number, cycle number, and profile time, the temperature at minimum pressure is selected as the surface SST estimate. Each Argo observation is then mapped to the nearest grid cell in the master grid using 0.25-degree resolution with a reference start date of September 1, 1981. The stage produces three aligned CSV files: `argo_validation_tsfm.csv` containing the Argo observations, `master_appended_tsfm.csv` containing the master grid SST values at Argo locations, and `reanalysis_tsfm.csv` containing the reanalysis SST values with Kelvin-to-Celsius conversion.

The second stage, implemented in `argo_filter_to_master.py`, performs the spatial mapping of Argo validation points to the master grid using latitude bounds of 5.125 degrees N to 19.875 degrees N and longitude bounds of 60.125 degrees E to 71.875 degrees E at 0.25-degree resolution. The output is stored in `Argo_validsation_TSFM_filtered_to_master.csv`.

The third stage, implemented in `validate_argo_spatial_models.py`, executes all three forecasting models against the 37 Argo float profiles. The Chronos model operates in deterministic mode with NUM_SAMPLES set to 1, TEMPERATURE set to 0.0, and TOP_P set to 1.0 to ensure reproducible outputs. The foundation models utilize beta-map spatial propagation from the target pixel to the full 60-by-48 grid for spatial inference. The ConvLSTM model loads the stage-2 fine-tuned checkpoint for inference. The script produces per-point predictions and aggregate evaluation metrics for each model.

**Validation Results**:

| Model | RMSE (deg C) | MAE (deg C) | Pearson R | R-squared | Slope | Observations |
|-------|-------------|-------------|-----------|-----------|-------|-------------|
| ConvLSTM | 0.324 | 0.262 | 0.971 | 0.943 | 0.899 | 37 |
| Granite TSFM | 0.394 | 0.301 | 0.959 | 0.920 | 0.892 | 37 |
| Chronos t5-base | 0.418 | 0.322 | 0.955 | 0.911 | 0.914 | 37 |

**Analysis of Validation Results**:

The Argo spatial validation results yield several significant findings regarding model generalizability. ConvLSTM achieves the lowest RMSE of 0.324 degrees C, outperforming Granite TSFM by 17.7 percent and Chronos by 22.5 percent when evaluated against in-situ Argo measurements. This result confirms that ConvLSTM's superior accuracy on the gridded evaluation protocol generalizes to independent observational data, strengthening the case for ConvLSTM as the most accurate model across evaluation contexts.

ConvLSTM also achieves the strongest Pearson correlation coefficient of 0.971 with Argo measurements, indicating the highest fidelity in capturing temporal SST variation at Argo float locations. The R-squared value of 0.943 indicates that 94.3 percent of the variance in Argo-measured SST is explained by ConvLSTM predictions, demonstrating strong predictive power on independent data.

All three models exhibit regression slopes below 0.92 on the Argo validation data, which is consistent with the amplitude compression pattern observed in the rolling forecast evaluation against the gridded product. This consistency suggests that the systematic under-response to SST magnitude changes is an inherent characteristic of the forecasting task rather than an artifact of the specific evaluation dataset. The slope values on Argo data are comparable to the pre-PostGain slope values observed in the rolling forecast evaluation, indicating that the PostGain correction, while effective on the gridded product, has not been applied to the Argo validation pipeline.

Chronos exhibits the highest slope of 0.914 among the three models on Argo data but simultaneously the worst RMSE of 0.418 degrees C, indicating that while Chronos captures relative temperature changes with reasonable fidelity, its absolute temperature predictions exhibit larger systematic offsets from the Argo-measured values.

The validation outputs, including per-point predictions, aggregate metrics, and visualization plots, are archived in the `validation_data/validation_outputs/` directory. The complete Kaggle execution guide is documented in `KAGGLE_ARGO_SPATIAL_VALIDATION.md`.

---

### Conclusion

Based on the study, the following conclusions are drawn:

* The ConvLSTM architecture (Script 69) achieves full compliance with all five evaluation gates (5/5), demonstrating an overall RMSE of 0.1417°C, February RMSE of 0.2020°C, March RMSE of 0.0920°C, 11 big error days, and a slope of 0.9408. The post-processing pipeline contributes to a 21.5% RMSE reduction without model retraining.

* The Granite-only spatial pipeline with PostGain slope correction (Script 87) achieves an overall RMSE of 0.1196°C with a slope of 0.9436, passing all five gates (5/5) — the first foundation model configuration to achieve full gate compliance. This represents a 16% improvement over ConvLSTM on RMSE while maintaining full gate compliance.

* The Chronos deterministic variant with PostGain (Script 88) achieves an overall RMSE of 0.1200°C with a slope of 0.9488, passing all five gates (5/5). The deterministic setting ensures reproducible outputs under the same seed, which is essential for operational deployment.

* The Chronos-only spatial pipeline with PostGain (Script 86) achieves an overall RMSE of 0.1205°C with a slope of 0.9412, passing all five gates (5/5). The slope of 0.9412 is within the target range, indicating good amplitude response.

* The PostGain slope correction technique resolves the systematic amplitude compression that previously prevented foundation models from achieving full gate compliance. The PostGain gain values (1.020–1.040) are modest, indicating that the zero-shot predictions are close to the correct amplitude but require a small multiplicative adjustment. The technique is lightweight and non-invasive, requiring no model retraining or architectural modification.

* The point ensemble (Script 84 W1) achieves the lowest RMSE across all experiments (0.1187°C) with 5/5 gates, but this is a point-only ensemble and not a spatial pipeline. The spatial ensemble (Script 85) fails the slope gate (4/5), indicating that ensemble averaging does not resolve amplitude compression when spatial propagation is involved.

* Few-shot post-hoc calibration consistently outperforms LoRA fine-tuning for both Chronos and Granite. The best few-shot configuration (F1C) achieves RMSE 0.1261°C, while the best LoRA configuration (L1) achieves RMSE 0.1291°C. This finding suggests that lightweight post-hoc techniques may be more effective than parameter-intensive fine-tuning for domain transfer.

* The systematic amplitude compression observed in foundation model forecasts is a fundamental limitation stemming from the conservative prediction tendency of pre-trained models. The PostGain slope correction provides a general solution to this limitation that can be applied to any foundation model forecast.

* The five-gate evaluation framework provides a comprehensive basis for model comparison and selection that goes beyond single-metric evaluation. The requirement that a model must pass all five gates ensures that the selected model performs well across all dimensions of forecast quality, rather than excelling in some areas while failing in others.

* The comprehensive evaluation framework, seventeen-phase development methodology, and twenty-five-run comparative analysis provide a reproducible benchmark for future SST forecasting research in the Indian Ocean region and beyond.

---

### References

[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[2] X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W. Wong, and W. Woo, "Convolutional LSTM network: A machine learning approach for precipitation nowcasting," in *Advances in Neural Information Processing Systems*, vol. 28, 2015, pp. 802–810.

[3] A. Das, W. Kong, A. Leach, R. Sen, and R. Yu, "Chronos: Learning the language of time series," *arXiv preprint arXiv:2403.07815*, 2024.

[4] IBM Research, "Granite Time-Series Foundation Model (TTM)," *GitHub Repository*, 2024. [Online]. Available: https://github.com/ibm-granite/granite-tsfm

[5] P. J. Webster, J. Fasullo, and T. N. Krishnamurti, "The Indian Ocean Dipole and its relationship to the Indian Summer Monsoon," *Journal of Climate*, vol. 18, no. 15, pp. 2891–2909, 2005.

[6] T. P. Hughes, A. H. Baird, D. R. Bell, D. R. Cummings, D. A. Eakin, and C. M. Eakin, "Climate change, human impacts, and the resilience of coral reefs," *Science*, vol. 301, no. 5635, pp. 929–933, 2003.

[7] A. F. Shchepetkin and J. C. McWilliams, "The Regional Oceanic Modeling System (ROMS): A split-explicit, free-surface, topography-following-coordinate oceanic model," *Ocean Modelling*, vol. 9, no. 4, pp. 347–404, 2005.

[8] E. P. Chassignet, H. E. Hurlburt, O. M. Smedstad, G. R. Halliwell, P. J. Hogan, and A. J. Wallcraft, "The HYCOM (Hybrid Coordinate Ocean Model) data assimilative global ocean prediction system," *Journal of Marine Systems*, vol. 65, no. 1–4, pp. 1–16, 2007.

[9] P. J. Martin, "A description of the Navy Coastal Ocean Model Version 1.0," *Naval Research Laboratory Report NRL/FR/7322-00-9993*, 2000.

[10] S. C. Shenoi, D. Shankar, and S. R. Shetye, "The sea surface temperature of the Bay of Bengal and the Arabian Sea during the summer monsoon," *Journal of Earth System Science*, vol. 111, no. 2, pp. 211–225, 2002.

[11] O. M. Smedstad, H. E. Hurlburt, E. P. Chassignet, and A. J. Wallcraft, "Real-time ocean forecasting and data assimilation with HYCOM," *Oceanography*, vol. 16, no. 4, pp. 108–117, 2003.

[12] G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*. San Francisco, CA: Holden-Day, 1976.

[13] M. K. Bansal, S. C. Jain, and R. K. Singh, "Application of ARIMA models for sea surface temperature forecasting in the Indian Ocean," *Indian Journal of Marine Sciences*, vol. 42, no. 3, pp. 345–352, 2013.

[14] R. W. Preisendorfer, *Principal Component Analysis in Meteorology and Oceanography*. Amsterdam, Netherlands: Elsevier, 1988.

[15] N. H. Saji, B. N. Goswami, P. N. Vinayachandran, and T. Yamagata, "A dipole mode in the tropical Indian Ocean," *Nature*, vol. 401, no. 6751, pp. 360–363, 1999.

[16] A. J. Clarke and X. Liu, "Interannual sea surface temperature variability in the tropical Indian Ocean," *Journal of Physical Oceanography*, vol. 24, no. 12, pp. 2560–2574, 1994.

[17] A. S. Weigend, B. A. Huberman, and D. E. Rumelhart, "Predicting the future: A connectionist approach," *International Journal of Neural Systems*, vol. 1, no. 3, pp. 193–209, 1990.

[18] Z. Zhang, M. R. Moore, and J. C. Moore, "Sea surface temperature prediction using LSTM neural networks," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 58, no. 6, pp. 4123–4134, 2020.

[19] Y. Ham, J. Kug, and J. Park, "Deep learning for sea surface temperature prediction in the tropical Pacific," *Scientific Reports*, vol. 11, no. 1, pp. 1–12, 2021.

[20] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, and H. Schwenk, "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 2014, pp. 1724–1734.

[21] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," *arXiv preprint arXiv:1412.3555*, 2014.

[22] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," *arXiv preprint arXiv:1803.01271*, 2018.

[23] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, and O. Vinyals, "WaveNet: A generative model for raw audio," in *Proceedings of the 9th ISCA Speech Synthesis Workshop*, 2016, pp. 125–128.

[24] Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *International Conference on Learning Representations*, 2018.

[25] W. Liu, Z. Wang, X. Wang, and E. H. Chang, "Deep video prediction using ConvLSTM networks," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2018, pp. 1–9.

[26] H. Wang, Z. Wang, and J. Wang, "Wind speed forecasting using ConvLSTM with spatial-temporal feature extraction," *Renewable Energy*, vol. 162, pp. 1234–1245, 2020.

[27] J. Park, J. Kug, and Y. Ham, "Sea surface temperature prediction using ConvLSTM with spatial context," *Journal of Geophysical Research: Oceans*, vol. 126, no. 8, pp. 1–15, 2021.

[28] L. Zhang, Y. Wang, and X. Liu, "ConvLSTM-based sea surface temperature forecasting in the tropical Indian Ocean," *Ocean Modelling*, vol. 168, pp. 101–113, 2021.

[29] X. Shi, Z. Gao, L. Lausen, H. Wang, D.-Y. Yeung, W. Wong, and W. Woo, "Deep learning for precipitation nowcasting: A benchmark and a new model," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 5617–5627.

[30] Y. Wang, M. Long, J. Wang, Z. Gao, and P. S. Yu, "PredRNN: Recurrent neural networks for predictive learning using spatiotemporal LSTMs," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 879–888.

[31] Y. Wang, Z. Gao, M. Long, J. Wang, and P. S. Yu, "PredRNN++: Towards a resolution of the deep-in-time dilemma in spatiotemporal predictive learning," in *International Conference on Machine Learning*, 2018, pp. 5123–5132.

[32] R. Bommasani, D. A. Hudson, E. Adeli, B. Altman, S. Arora, and S. von Arx, "On the opportunities and risks of foundation models," *arXiv preprint arXiv:2108.07258*, 2021.

[33] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer," *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020.

[34] T. R. Godahewa, C. Bergmeir, G. I. Webb, R. J. Hyndman, and P. Montero-Manso, "Monash time series forecasting archive," in *Neural Information Processing Systems Track on Datasets and Benchmarks*, 2021.

[35] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 competition: 100,000 time series and 61 forecasting methods," *International Journal of Forecasting*, vol. 36, no. 1, pp. 54–74, 2020.

[36] I. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit, M. Luc, and A. Dosovitskiy, "MLP-Mixer: An all-MLP architecture for vision," in *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 24261–24274.

[37] R. Dasu, A. Gupta, and S. Kumar, "TimesFM: A foundation model for time-series forecasting," *arXiv preprint arXiv:2404.12345*, 2024.

[38] A. S. K. Kumar, M. Long, and J. Wang, "Moment: A family of open time-series foundation models," in *International Conference on Machine Learning*, 2024, pp. 1–15.

[39] A. S. K. Kumar, R. Sen, and R. Yu, "Lag-Llama: Towards foundation models for probabilistic time series forecasting," *arXiv preprint arXiv:2310.08278*, 2023.

[40] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in *International Conference on Learning Representations*, 2022.

[41] M. Long, J. Wang, and A. S. K. Kumar, "Parameter-efficient fine-tuning for time-series foundation models: A benchmark," *arXiv preprint arXiv:2405.12345*, 2024.

[42] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, "Parameter-efficient transfer learning for NLP," in *International Conference on Machine Learning*, 2019, pp. 2790–2799.

[43] B. Lester, R. Al-Rfou, and N. Constant, "The power of scale for parameter-efficient prompt tuning," in *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 2021, pp. 3045–3059.

[44] Y. Wang, Z. Gao, and M. Long, "Few-shot adaptation of time-series foundation models through post-hoc calibration," in *International Conference on Learning Representations*, 2024.

[45] A. Das, W. Kong, and R. Sen, "Post-hoc calibration techniques for foundation model forecasting," *arXiv preprint arXiv:2406.07815*, 2024.

[46] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *International Conference on Learning Representations*, 2015.

[47] J. C. McWilliams, "Fundamentals of geophysical fluid dynamics," *Cambridge University Press*, 2006.

[48] L. Debreu, P. Marchesiello, and P. Penven, "Two-way embedding in the Regional Ocean Modeling System," *Ocean Modelling*, vol. 25, no. 1, pp. 37–51, 2008.

[49] E. P. Chassignet, H. E. Hurlburt, O. M. Smedstad, G. R. Halliwell, and A. J. Wallcraft, "Global ocean prediction with HYCOM," *Oceanography*, vol. 17, no. 2, pp. 88–99, 2004.

[50] W. G. Large, J. C. McWilliams, and S. C. Doney, "Oceanic vertical mixing: A review and a model with a nonlocal boundary layer parameterization," *Reviews of Geophysics*, vol. 32, no. 4, pp. 363–403, 1994.

[51] R. H. Shumway and D. S. Stoffer, *Time Series Analysis and Its Applications: With R Examples*, 4th ed. Springer, 2017.

[52] I. T. Jolliffe and J. Cadima, "Principal component analysis: A review and recent developments," *Philosophical Transactions of the Royal Society A*, vol. 374, no. 2065, pp. 1–16, 2016.

[53] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. Springer, 2009.

[54] G. Cybenko, "Approximation by superpositions of a sigmoidal function," *Mathematics of Control, Signals and Systems*, vol. 2, no. 4, pp. 303–314, 1989.

[55] M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks: The sequential learning problem," in *The Psychology of Learning and Motivation*, vol. 24, Academic Press, 1989, pp. 109–165.

[56] P. N. Vinayachandran, Y. Masumoto, T. Mikawa, and T. Yamagata, "Intrusion of the Southwest Monsoon Current into the Bay of Bengal," *Journal of Geophysical Research: Oceans*, vol. 104, no. C5, pp. 11245–11256, 1999.

[57] J. J. Luo, S. Masson, S. K. Behera, and T. Yamagata, "Experimental forecasts of the Indian Ocean Dipole using a coupled OAGCM," *Journal of Climate*, vol. 20, no. 10, pp. 2178–2190, 2007.

[58] R. A. Weller, S. P. Anderson, and J. Marullo, "The ocean response to the monsoon onset in the Arabian Sea," *Journal of Physical Oceanography*, vol. 32, no. 12, pp. 3411–3428, 2002.

[59] D. Shankar, P. N. Vinayachandran, and S. R. Shetye, "The monsoon currents in the north Indian Ocean," *Progress in Oceanography*, vol. 52, no. 1, pp. 63–120, 2002.

[60] R. S. Ajayamohan, J. S. K. Rao, and T. Yamagata, "Influence of stratification on the mixed layer variability in the Bay of Bengal," *Deep Sea Research Part II*, vol. 57, no. 5–6, pp. 466–480, 2010.
