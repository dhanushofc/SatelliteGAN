# Why We Stopped Arguing and Started Building: A Foundation Model Approach to Time Series Forecasting

There is a familiar pattern in time series teams. Someone proposes a new architecture. Someone else defends the old one. Weeks are spent debating window sizes, normalization strategies, and whether the validation split should be 80/20 or 85/15. The arguments are earnest, the whiteboards fill up, and the model stays in development.

We found ourselves in that pattern. Then we stopped.

The decision was not ideological. It was practical. We were working on a sea surface temperature forecasting problem with 16,000+ days of gridded oceanographic data, and every custom architecture we built — ConvLSTM variants, multi-horizon strategies, branching ensembles — required weeks of hyperparameter tuning, careful data pipeline engineering, and fragile post-processing logic that broke whenever the input distribution shifted slightly. The models worked, but the cost of keeping them working was high.

Around the same time, time series foundation models were maturing. Models like Chronos and Granite TTM had been pre-trained on thousands of diverse time-series datasets, learning representations that generalized across domains. The claim was not that they would outperform a perfectly tuned custom model on every task. The claim was that they would get you close enough, fast enough, with far less effort. We decided to test that claim.

The approach was straightforward. We ran the foundation models in zero-shot mode — no fine-tuning, no domain-specific training — then applied a lightweight post-hoc correction pipeline using our validation set. The correction consisted of three steps: a Ridge regression on the residuals, an amplitude calibration to address systematic under-prediction, and an adaptive drift correction to handle temporal bias. The entire pipeline fit in a single script. No architecture search. No hyperparameter grid. No weeks of tuning.

In our experiments, the foundation model approach was consistently within a narrow margin of the best tuned custom pipeline, while requiring a fraction of the development time. More importantly, the foundation models generalized better to unseen conditions. When the monsoon transition period introduced dynamics that our ConvLSTM had not been trained on, the foundation model handled it without degradation. We suspect this generalizability arises from the diversity of the pre-training corpus — these models have seen regime transitions before, even if not in oceanographic data.

The efficiency gains were notable. Training time dropped from hours to minutes. The number of hyperparameter trials decreased by an order of magnitude. The post-processing pipeline, which had grown to four stages of increasingly fragile corrections, was replaced by a single gain multiplier fitted on validation data. The entire system became easier to reason about, easier to reproduce, and easier to deploy.

This is not an argument against custom architectures. There are tasks where a purpose-built model is the right choice. But for many time series problems, the marginal accuracy gain from weeks of tuning does not justify the engineering debt it creates. Foundation models offer a different trade-off: slightly less peak performance in exchange for dramatically lower complexity, faster iteration, and better generalization.

The lesson we took from this experience was not that foundation models are inherently superior. It was that the energy spent defending legacy approaches or debating minor methodological choices is often better invested in trying something simpler. We stopped arguing about window sizes and normalization strategies. We ran the foundation model. We applied a lightweight correction. We checked the results. They were good enough to ship.

Sometimes the best methodological decision is the one that lets you stop debating and start building.
