from fastapi import FastAPI
from inference import main

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

# This is REQUIRED for hackathon
def main_entry():
    main()

# Hackathon expects this name
main = main_entry