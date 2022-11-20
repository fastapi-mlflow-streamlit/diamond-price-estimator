# Diamond Price Estimator

A machine learning project made with PyCaret, FastAPI and Gradio.

The diamond.py file contains a FastAPI app, which displays a Gradio interface at the /diamond route. The interface allows the user to input a series of features (carat weight, cut, color, clarity, polish, etc.) and the app then outputs a price prediction by employing a regression model (Diamond_Estimator_LightGBM.pkl). The Diamond_pycaret_model.ipynb notebook illustrates the procedure that was carried out to train the model with PyCaret.
