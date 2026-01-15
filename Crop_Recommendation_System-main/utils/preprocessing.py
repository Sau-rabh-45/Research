# preprocessing.py
import numpy as np

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def prepare_input(form):
    return np.array([[
        float(form["N"]),
        float(form["P"]),
        float(form["K"]),
        float(form["temperature"]),
        float(form["humidity"]),
        float(form["ph"]),
        float(form["rainfall"]),
    ]])