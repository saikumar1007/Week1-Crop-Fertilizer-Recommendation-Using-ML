# Week1-Crop-Fertilizer-Recommendation-Using-ML
This project uses machine learning to analyze soil type, climate, and nutrient levels, predicting the best crop to grow and recommending suitable fertilizers to optimize agricultural productivity.
README - Crop Recommendation System Notebook
============================================

This Jupyter Notebook (crop_fixed.ipynb) processes agricultural data to prepare it for a crop recommendation system.

Steps in the Notebook:
----------------------
1. **Import Libraries**:
   - Imports required Python libraries: numpy, pandas, seaborn, matplotlib, and sklearn.

2. **Load Dataset**:
   - Reads the crop recommendation dataset from:
     /home/user/CROP AND FERTILIZER RS/dataset/Crop_recommendation.csv

3. **Preprocess Data**:
   - Drops the 'label' column to isolate features.
   - Applies `StandardScaler` to normalize the feature values.

4. **Save Artifacts**:
   - Saves the fitted scaler to a file named `scaler.pkl`.
   - (Optionally) Saves the model `DT` to `model.pkl` — ensure `DT` is defined earlier in the notebook.

How to Use:
-----------
- Make sure the dataset exists at the specified path.
- If using your own model, define it as `DT` before the save block.
- Run all cells in order.
- You can later load the scaler/model using `pickle.load()` for inference.

File List:
----------
- `crop_fixed.ipynb`: Fixed Jupyter Notebook.
- `scaler.pkl`: Saved StandardScaler object (after running the notebook).
- `README.txt`: This file.

To Run :
---------
- streamlit run app.py
