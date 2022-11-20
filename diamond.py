import uvicorn
from fastapi.responses import RedirectResponse
from fastapi import FastAPI
import gradio as gr
from pycaret.regression import load_model, predict_model
import pandas as pd


# the gradio interface will be displayed at the /diamond route
CUSTOM_ROUTE = "/diamond"

app = FastAPI()


# DIAMOND ESTIMATOR FUNCTION

# see Diamond_pycaret_model.ipynb for the model training
model = load_model('Diamond_Estimator_LightGBM')

def predict(carat_weight, cut, color, clarity, polish, symmetry, report):
    # prepare pandas DataFrame with the inputs from the gradio interface
    inputs_df = pd.DataFrame([[carat_weight, cut, color, clarity, polish, symmetry, report]])
    inputs_df.columns = ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']
    # make prediction
    prediction = predict_model(model, data=inputs_df)
    # return only the number
    return round(prediction['Label'][0], 2)


# ROUTES

# The home route automatically redirects to /diamond
@app.get("/")
def home():
    return RedirectResponse(CUSTOM_ROUTE)


# GRADIO INTERFACE

io = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label='Carat Weight'),
        gr.Dropdown(['Signature-Ideal', 'Ideal', 'Very Good', 'Good', 'Fair'], label='Cut'),
        gr.Dropdown(['D', 'E', 'F', 'G', 'H', 'I'], label='Color'),
        gr.Dropdown(['F', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1'], label='Clarity'),
        gr.Dropdown(['ID', 'EX', 'VG', 'G'], label='Polish'),
        gr.Dropdown(['ID', 'EX', 'VG', 'G'], label='Simmetry'),
        gr.Dropdown(['AGSL', 'GIA'], label='Report')
    ],
    outputs=[gr.Number(label='Price Prediction in USD')],
    allow_flagging='never',
    title="Diamond Price Estimator",
    description="""
    Legenda
    * Carat Weight: The weight of the diamond in metric carats. One carat is equal to 0.2 grams.
    * Cut: One of five values indicating the cut of the diamond in the following order of desirability 
        (Signature-Ideal, Ideal, Very Good, Good, Fair).
    * Color: One of six values indicating the diamond's color in the following order of desirability 
        (D, E, F, G, H, I).
    * Clarity: One of seven values indicating the diamond's clarity in the following order of desirability 
        (F = Flawless, IF = Internally Flawless, VVS1 or VVS2 = Very, Very Slightly Included, VS1 or VS2 = Very Slightly Included, SI1 = Slightly Included).
    * Polish: One of four values indicating the diamond's polish 
        (ID = Ideal, EX = Excellent, VG = Very Good, G = Good).
    * Symmetry: One of four values indicating the diamond's symmetry 
        (ID = Ideal, EX = Excellent, VG = Very Good, G = Good).
    * Report: One of of two values "AGSL" or "GIA" indicating which grading agency reported the qualities of the diamond qualities.
    """
)
# mount gradio interface
# method 1:
# gradio_app = gr.routes.App.create_app(io)
# app.mount(CUSTOM_ROUTE, gradio_app)
# method 2:
app = gr.mount_gradio_app(app, io, path=CUSTOM_ROUTE)


# run with: uvicorn thisfilename:app
if __name__=="__main__":
    # uvicorn.run("thisfilename:app")
    uvicorn.run("diamond:app")