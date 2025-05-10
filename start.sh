#!/bin/bash
uvicorn energy_prediction_api:app --host=0.0.0.0 --port=10000
# --reload
# --reload is used for development purposes only
# --host=