from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models

app = FastAPI()

class ContentRequest(BaseModel):
    content: str

@app.post("/generate-content")
async def generate_content(
    request: ContentRequest,
    max_output_tokens: int = Query(2048, description="Maximum number of output tokens"),
    temperature: float = Query(0.9, description="Temperature for content generation"),
    top_p: float = Query(1, description="Top-p value for content generation"),
):
    vertexai.init(project="colakin-dev-platform", location="us-central1")
    model = GenerativeModel("gemini-1.0-pro-001")

    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    responses = model.generate_content(
        request.content,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    async def generate():
        for response in responses:
            yield response.text

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)