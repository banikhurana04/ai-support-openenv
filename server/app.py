from fastapi import FastAPI
from environment import SupportTicketEnv
from models import Action

app = FastAPI()
env = SupportTicketEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, info = env.step(act)
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info
    }

# ✅ Hackathon entry point
def main():
    obs = env.reset()
    return obs.model_dump()

if __name__ == "__main__":
    print(main())