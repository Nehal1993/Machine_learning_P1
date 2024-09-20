from fastapi import FastAPI
import joblib
import gradio as gr
import uvicorn
import threading
import time
import nest_asyncio

# Apply nest_asyncio to avoid asyncio issues
nest_asyncio.apply()

# Load the model
Loaded_model = joblib.load('fuel_cost_6000_miles.sav')

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Welcome to Our Fuel Price Prediction machine learning model"}

@app.get("/instructions")
async def instructions():
    return {
        "message": (
            "Year range of our model is 2000-2023, "
            "Euro Standard range is 3.0-5.0, "
            "Engine Capacity is from 600-3500, "
            "Combined Metrices is 6.0-10.0, "
            "Combined Imperial is from 20.0-50, "
            "Enter 1 if it matches your fuel type otherwise enter 0."
        )
    }

def predict(year, Euro_standard, Engine_Capacity, Co2_emission, Combined_metrices, Combined_Imperial, Diesel, LPG, petrol):
    try:
        # Prediction returns a float
        prediction = Loaded_model.predict([[year, Euro_standard, Engine_Capacity, Co2_emission,
                                            Combined_metrices, Combined_Imperial, Diesel, LPG, petrol]])
        return f"£{float(prediction):.2f}"
    except Exception as e:
        return str(e)

# Gradio interface with components
def run_gradio():
    gr_interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Slider(minimum=2000, maximum=2023, step=1, label="Year"),
            gr.Slider(minimum=3.0, maximum=5.0, step=0.1, label="Euro Standard"),
            gr.Slider(minimum=600, maximum=3500, step=10, label="Engine Capacity"),
            gr.Slider(minimum=0, maximum=500, step=1, label="Co2 Emission"),
            gr.Slider(minimum=6.0, maximum=10.0, step=0.1, label="Combined Metrices"),
            gr.Slider(minimum=20.0, maximum=50.0, step=0.1, label="Combined Imperial"),
            gr.Checkbox(label="Diesel"),
            gr.Checkbox(label="LPG"),
            gr.Checkbox(label="Petrol")
        ],
        outputs="text"
    )
    gr_interface.launch(server_name="0.0.0.0", server_port=7860, share=True)

def start_gradio():
    gradio_thread = threading.Thread(target=run_gradio)
    gradio_thread.start()
    time.sleep(5)  # Ensure Gradio has time to start

# Start FastAPI app and Gradio interface
if __name__ == "__main__":
    # Start Gradio first
    start_gradio()
    # Then start FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
