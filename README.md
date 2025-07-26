There are four main steps (now postfact explained):
1. Understanding the data, that can be done by `uv run data_analysis.py`  
2. Deciding the best preprocessing steps, that can be done by `uv run stft_analysis.py`
3. Training via `uv run train_naive.py`
4. Predict for the private testset via `uv run inference.py`