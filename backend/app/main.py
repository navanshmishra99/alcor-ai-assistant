from fastapi import FastAPI

app = FastAPI(title="Alcor AI Assistant")


@app.get("/health")
def health_check():
    return {"status": "ok"}