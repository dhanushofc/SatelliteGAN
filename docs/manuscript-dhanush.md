# IEEE-Style Research Manuscript

## Document Identification
* **Title**: Sea Surface Temperature Forecasting Using Convolutional LSTM and Foundation Models with PostGain Slope Correction
* **Authors**: G. Dhanush
* **Affiliations**: Department of Computer Science and Engineering, ICFAI Foundation for Higher Education (IFHE)
* **Context**: Research Internship, Indian National Centre for Ocean Information Services (INCOIS)

---

### Abstract

Accurate sea surface temperature forecasts are critical for operational oceanography, monsoon prediction, and marine ecosystem management in the Indian Ocean region. This study presents a comprehensive comparative evaluation of three distinct forecasting paradigms — a custom Convolutional Long Short-Term Memory architecture, Amazon Chronos (a transformer-based foundation model), and IBM Granite Time-Series Foundation Model — applied to 7-day rolling SST prediction over a 60 by 48 spatial grid spanning 16,290 days of observations. A five-gate evaluation framework was established assessing overall accuracy, seasonal performance, extreme event handling, and amplitude fidelity. The ConvLSTM architecture achieved full gate compliance with an RMSE of 0.1417 degrees Celsius. Foundation models in their native zero-shot and few-shot configurations consistently failed the slope gate, exhibiting systematic amplitude compression. A novel PostGain slope correction technique was developed that applies a post-hoc gain multiplier fitted on validation data, resolving this limitation. The Granite-only PostGain configuration achieved an RMSE of 0.1196 degrees Celsius with a slope of 0.9436, passing all five gates and becoming the first foundation model to achieve full compliance. Independent validation against 37 Argo float profiles confirmed ConvLSTM achieved the strongest in-situ correlation (R equals 0.971, RMSE 0.324 degrees Celsius). The PostGain technique provides a lightweight, retraining-free solution for operational deployment of foundation models in oceanographic forecasting.

**Keywords**—Sea Surface Temperature, ConvLSTM, Chronos, Granite TTM, PostGain Slope Correction, Foundation Models, Time Series Forecasting, Indian Ocean.

---

## I. Introduction

Sea surface temperature serves as a fundamental parameter in oceanographic science, directly modulating air-sea heat exchange, monsoon dynamics, marine ecosystem productivity, and tropical cyclone intensity. The Indian Ocean region, bounded by latitudes 5 degrees North to 20 degrees North and longitudes 60 degrees East to 72 degrees East, encompasses the Laccadive Sea and the Arabian Sea, where SST variability drives the Southwest Monsoon that sustains the agricultural economy of over one billion people. Operational forecasting centers such as the Indian National Centre for Ocean Information Services require reliable 7-day SST predictions for fisheries management, navigation safety, and climate monitoring.

Traditional approaches to SST forecasting have followed two primary paths. Numerical Weather Prediction models such as the Regional Ocean Modeling System and the Hybrid Coordinate Ocean Model solve coupled partial differential equations governing ocean dynamics, achieving physically consistent forecasts at substantial computational cost [1], [2]. Statistical methods including autoregressive models and empirical orthogonal function analysis offer computational efficiency but assume linear dynamics that do not hold for tropical SST variability.

The emergence of deep learning has transformed time-series forecasting. Long Short-Term Memory networks demonstrated superior temporal modeling capability over statistical methods [3]. Convolutional LSTM architectures extended this by integrating spatial convolution within the LSTM cell, enabling simultaneous capture of spatial and temporal dependencies in gridded data [4]. More recently, foundation models pre-trained on diverse multi-domain corpora — Amazon Chronos [5] and IBM Granite Time-Series Foundation Model [6] — have demonstrated remarkable zero-shot forecasting capabilities across domains.

Despite their promise, foundation models applied to SST forecasting exhibit a critical limitation: systematic amplitude compression. While these models accurately predict the direction of temperature change, they systematically under-predict the magnitude, failing the slope gate of the evaluation framework. Previous attempts to address this through LoRA fine-tuning [7], post-hoc calibration, and adaptive drift correction achieved only marginal improvements (best slope 0.9253).

This work contributes three novel advances. First, a comprehensive five-gate evaluation framework that assesses forecast quality across multiple dimensions simultaneously. Second, the PostGain slope correction technique — a post-hoc gain multiplier fitted on validation data that resolves amplitude compression without model retraining. Third, independent validation against 37 in-situ Argo float profiles, providing the first assessment of foundation model generalizability at Argo observation points in the Indian Ocean.

---

## II. Data and Methodology

The methodology is structured around a single unifying narrative: starting from raw observational data, three modeling paradigms were developed and refined through a structured evaluation pipeline, and the best-performing configurations were independently validated against in-situ measurements.

### A. Dataset Specification

The Optimum Interpolation Sea Surface Temperature (OISST) dataset, processed by INCOIS into a daily gridded product, forms the observational foundation of this study. The spatial domain spans latitudes 5 degrees North to 20 degrees North and longitudes 60 degrees East to 72 degrees East at 0.25 degree resolution, yielding a 60 by 48 pixel grid. The temporal domain covers 16,290 daily observations from September 1, 1981 to April 7, 2026.

The target prediction point is located at 8.0 degrees North, 67.0 degrees East in the Laccadive Sea. The data was partitioned into a training set comprising 85 percent of observations (September 1, 1981 to approximately April 2023), a validation set comprising 5 percent (approximately May 2023 to December 2023), and a test set comprising 10 percent corresponding to the 90-day evaluation period from January 1, 2026 to March 31, 2026. All normalization statistics were computed exclusively on the training partition to prevent data leakage.

Long-term daily means were computed from the full training period and subtracted from absolute SST to produce anomaly fields. The anomaly fields were normalized using training-set mean and standard deviation. For the foundation model pipelines, the anomaly sequence was padded to 60-day windows with an input length of 64 time steps and a forecast horizon of 7 days.

### B. Architectural Framework and Custom Configurations

Three distinct modeling architectures were evaluated. The first architecture is a purpose-built ConvLSTM comprising two ConvLSTM cells with a hidden dimension of 64, followed by a convolutional neck and a convolutional head. The model processes 60-day input sequences over the full 60 by 48 spatial grid. Input channels include the normalized SST anomaly, the long-term daily mean, latitude coordinates, and longitude coordinates. The ConvLSTM cell structure follows the formulation of Shi et al. [4], where the input, forget, output, and cell gates are computed through convolutional operations that preserve the two-dimensional spatial structure. Post-processing for the ConvLSTM includes per-horizon bias correction, adaptive drift correction with a 7-day window capped at plus or minus 0.20 degrees Celsius, and 5-day rolling mean smoothing. The training was conducted on an NVIDIA T4 GPU for 15 to 25 epochs with a batch size of 8.

The second architecture is Amazon Chronos, specifically the chronos-t5-base variant with 200 million parameters. Chronos is a transformer-based foundation model pre-trained on thousands of diverse time-series datasets. This study deploys Chronos in a zero-shot configuration where the pre-trained weights remain frozen. A per-horizon bias correction is computed from the 689 validation windows, followed by a Ridge residual corrector fitted independently for each of the seven forecast horizons. Amplitude calibration applies per-horizon scaling factors clipped to the range 0.85 to 1.00. The critical addition is the PostGain slope targeter, which fits a post-hoc gain multiplier on validation data to achieve a slope of at least 0.94. The PostGain gain for Chronos is 1.040. For spatial inference, a beta-map is computed from the correlation between the target pixel and each grid cell across the training set, enabling propagation of the point forecast to the full 60 by 48 spatial field.

The third architecture is IBM Granite Time-Series Foundation Model, specifically the Granite-TTM-r2 variant with 71,000 parameters. Granite employs an MLP-Mixer-based architecture pre-trained on a diverse corpus of time-series data. The deployment pipeline is identical to Chronos: zero-shot inference with frozen weights, per-horizon bias correction, Ridge residual correction, amplitude calibration, and the PostGain slope targeter. The PostGain gain for Granite is 1.020, reflecting the closer baseline slope of the Granite zero-shot predictions.

Additional configurations were evaluated for comparison. The few-shot configuration (Chronos F1C, Granite G1A) replaces the PostGain targeter with an adaptive drift correction using a window of 5 days and caps of plus or minus 0.20 degrees Celsius. The LoRA fine-tuning configuration applies low-rank adaptation matrices of rank 8 to the transformer attention layers, trained for 10 to 15 epochs on the SST training data.

### C. Optimization and Performance Gates

A rigorous five-gate evaluation framework was established to assess model performance across multiple dimensions simultaneously. A model must pass all five gates to be considered operationally compliant. Gate one is the overall RMSE threshold of 0.1466 degrees Celsius. Gate two is the February RMSE threshold of 0.2093 degrees Celsius, capturing the most challenging monsoon transition period. Gate three is the March RMSE threshold of 0.1003 degrees Celsius. Gate four is the big error count threshold of 12 days with absolute error exceeding 0.20 degrees Celsius. Gate five is the slope target range of 0.94 to 1.00, measuring the fidelity of amplitude response.

The slope metric is defined as the slope of the linear regression between predicted and observed SST values. A slope below 0.94 indicates systematic under-prediction of temperature magnitude, which has direct operational implications for extreme event detection and marine warning systems.

---

## III. Results and Discussion

### A. Post-Game Evaluation Analysis

The evaluation was conducted over a 90-day rolling forecast period from January 1, 2026 to March 31, 2026, comprising 98 overlapping forecast windows. For each window, the model ingests 60 days of historical SST anomalies and produces 7-day ahead forecasts. The post-game evaluation computes per-horizon and aggregate metrics comparing predicted values against the observed SST at the target location (8.0 degrees North, 67.0 degrees East).

The PostGain slope correction was identified as the critical intervention that enabled foundation models to achieve full gate compliance. The gain multiplier (1.020 for Granite, 1.040 for Chronos) is modest, indicating that the zero-shot predictions are close to the correct amplitude but systematically attenuated. The correction is stable across the evaluation period, with no evidence of over-correction or instability at extreme values.

Independent Argo float validation was conducted to assess generalizability beyond the OISST-derived evaluation product. Thirty-seven Argo float profiles spanning January to February 2026 were matched to the nearest grid cell. SST was extracted at minimum pressure per profile, with rigorous quality control filtering. All three models were evaluated at these spatial locations using beta-map propagation for the foundation models.

### B. Metric Performance Comparison

The following table presents the core metric performance for all evaluated configurations:

| Model Configuration Type | MSE | RMSE (degrees C) | R-Squared (R squared) | Correlation (r) | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ConvLSTM 69 (Baseline) | 0.0201 | 0.1417 | 0.788 | 0.888 | 5/5 Passed |
| Chronos Few-Shot F1C | 0.0159 | 0.1261 | — | — | 4/5 Failed Slope |
| Granite Few-Shot G1A | 0.0162 | 0.1272 | — | — | 4/5 Failed Slope |
| **Granite 87 PostGain** | **0.0143** | **0.1196** | **0.882** | **0.939** | **5/5 Exceeded** |
| Chronos 88 PostGain Det | 0.0144 | 0.1200 | 0.882 | 0.939 | 5/5 Met Threshold |
| Chronos 86 PostGain | 0.0145 | 0.1205 | 0.880 | 0.938 | 5/5 Met Threshold |

The PostGain-corrected configurations occupy the top three ranks. Granite 87 achieves the best single-model result with a 16 percent RMSE improvement over the ConvLSTM baseline while maintaining full gate compliance — the first foundation model to do so.

The monthly breakdown reveals important temporal patterns. Granite 87 achieves the lowest March RMSE (0.0857 degrees Celsius) and ties for the fewest big error days (9). Chronos 88 achieves the lowest February RMSE (0.1640 degrees Celsius), indicating superior performance during the monsoon transition period. The ConvLSTM baseline maintains competitive March RMSE (0.0920 degrees Celsius) and the strongest slope among pre-PostGain configurations (0.9408).

The independent Argo spatial validation results are presented below:

| Model | RMSE (degrees C) | MAE (degrees C) | Pearson R | R-Squared | Slope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ConvLSTM** | **0.324** | **0.262** | **0.971** | **0.943** | 0.899 |
| Granite TSFM | 0.394 | 0.301 | 0.959 | 0.920 | 0.892 |
| Chronos t5-base | 0.418 | 0.322 | 0.955 | 0.911 | 0.914 |

ConvLSTM achieves the lowest RMSE (0.324 degrees Celsius) among all three models when validated against in-situ Argo measurements, outperforming Granite by 17.7 percent and Chronos by 22.5 percent. This confirms that ConvLSTM's superior accuracy on the OISST evaluation product generalizes to independent observational data. All three models exhibit slopes below 0.92 on Argo data, consistent with the amplitude compression pattern observed in the OISST evaluation.

### C. Figure Asset and Image Formatting Guidelines

All figures in this manuscript should be rendered at a minimum resolution of 300 DPI and stored in a dedicated figures directory. The following caption structure is recommended:

**Fig. 1.** Ninety-day time series overlay comparing predicted SST against ground truth for ConvLSTM 69, Granite 87, Chronos 88, and Chronos 86 at the target location 8.0 degrees North, 67.0 degrees East.

**Fig. 2.** Scatter plot of predicted versus observed SST for the Granite 87 PostGain configuration, showing correlation R of 0.939 and slope of 0.9436.

**Fig. 3.** Argo spatial validation scatter plot comparing model predictions against in-situ Argo float measurements across 37 validation points.

**Fig. 4.** PostGain correction analysis showing the effect of the gain multiplier on slope and RMSE.

---

## IV. Conclusion and Future Work

This study demonstrates that foundation models for time-series forecasting, when equipped with a lightweight PostGain slope correction, can achieve full operational compliance for SST forecasting in the Indian Ocean region. The Granite 87 configuration achieved an RMSE of 0.1196 degrees Celsius with a slope of 0.9436, passing all five evaluation gates and representing a 16 percent improvement over a purpose-built ConvLSTM architecture. The PostGain technique requires no model retraining, making it directly applicable to operational deployment scenarios.

The independent Argo validation confirms that ConvLSTM achieves the strongest generalization to in-situ measurements, with the lowest RMSE (0.324 degrees Celsius) and the highest correlation (0.971) across 37 independent Argo float profiles. This finding establishes ConvLSTM as the most reliable model for operational use when absolute temperature accuracy at observation points is critical.

Several directions for future research emerge from this work. First, the PostGain technique could be extended to a dynamic formulation where the gain multiplier adapts to seasonal and interannual variability. Second, the Argo validation pipeline could be expanded to include a larger number of profiles across multiple years, enabling robust statistical assessment of spatial generalizability. Third, the evaluation framework could be extended to include multi-modal inputs such as wind stress, sea level pressure, and ocean current fields. Fourth, real-time streaming inference pipelines could be developed for operational deployment at INCOIS. Finally, the PostGain methodology may generalize to other geophysical forecasting tasks where foundation models exhibit systematic amplitude compression.

---

## References

[1] A. F. Shchepetkin and J. C. McWilliams, "The regional oceanic modeling system (ROMS): A split-explicit, free-surface, topography-following-coordinate oceanic model," *Ocean Modelling*, vol. 9, no. 4, pp. 347-404, 2005.

[2] R. Bleck, "An oceanic general circulation model framed in hybrid isopycnic-Cartesian coordinates," *Ocean Modelling*, vol. 4, no. 1, pp. 55-88, 2002.

[3] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.

[4] X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-C. Woo, "Convolutional LSTM network: A machine learning approach for precipitation nowcasting," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2015, pp. 802-810.

[5] A. Ansari, L. Tiao, A. Katharopoulos, et al., "Chronos: Learning the language of time series," *arXiv preprint arXiv:2403.07815*, 2024.

[6] IBM Research, "Granite time-series foundation model," IBM TTM Documentation, 2024.

[7] E. Hu, Y. Shen, P. Wallis, et al., "LoRA: Low-rank adaptation of large language models," in *Proc. International Conference on Learning Representations (ICLR)*, 2022.

[8] R. W. Reynolds, T. M. Smith, C. Liu, D. B. Chelton, K. S. Casey, and M. G. Schlax, "Daily high-resolution-blended analyses for sea surface temperature," *Journal of Climate*, vol. 20, no. 22, pp. 5473-5496, 2007.

[9] D. Roemmich, S. Riser, R. Davis, and L. Talley, "The Argo program: Observing the global ocean with profiling floats," *Oceanography*, vol. 22, no. 2, pp. 34-43, 2009.

[10] Indian National Centre for Ocean Information Services, "Operational oceanographic services," INCOIS, Ministry of Earth Sciences, Government of India, 2025.
