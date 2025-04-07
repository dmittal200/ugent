from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModel, AutoTokenizer
import requests
import nltk
from typing import Dict, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt_tab')  # Updated resource for tokenization
nltk.download('averaged_perceptron_tagger')

app = FastAPI(
    title="Dynamic Model Selection Agent",
    description="API to interpret user prompts and dynamically select Hugging Face models",
    version="1.0.0"
)

# Define request model
class UserPrompt(BaseModel):
    prompt: str

class ModelAgent:
    def __init__(self):
        self.loaded_models: Dict[str, any] = {}
        self.hf_api_url = "https://huggingface.co/api/models"
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.task_definitions = {
            "text-classification": ["classify", "sentiment", "positive", "negative", "emotion", "categorize", "predict labels"],
            "question-answering": ["question", "answer", "what", "how", "why", "explain", "tell me"],
            "text-generation": ["generate", "create", "write", "compose", "story", "article", "poem"],
            "summarization": ["summarize", "summary", "brief", "condense", "short version"],
            "translation": ["translate", "language", "convert to", "in spanish", "in french"],
            "named-entity-recognition": ["extract", "identify", "entities", "names", "locations"],
            "text-to-image": ["image", "picture", "generate image", "visualize"],
        }

    def analyze_prompt(self, prompt: str) -> Tuple[str, float]:
        """Analyze the prompt to determine the task type and confidence score"""
        # Tokenize the prompt
        prompt_lower = prompt.lower()
        tokens = nltk.word_tokenize(prompt_lower)

        # Define candidate labels (task types)
        candidate_labels = list(self.task_definitions.keys())

        # Use zero-shot classification to score tasks
        result = self.classifier(prompt, candidate_labels, multi_label=False)
        top_task = result["labels"][0]  # Most likely task
        confidence = result["scores"][0]  # Confidence score

        # Fallback to keyword-based check if confidence is low (<0.5)
        if confidence < 0.5:
            for task, keywords in self.task_definitions.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    return task, 0.9  # High confidence for keyword match
            return "text-classification", 0.1  # Default with low confidence
        
        return top_task, confidence

    def search_huggingface_models(self, task_type: str) -> str:
        """Search Hugging Face for a suitable model"""
        try:
            # Query Hugging Face API with task type
            params = {
                "pipeline_tag": task_type,
                "sort": "downloads",  # Sort by popularity
                "limit": 1  # Get top result
            }
            response = requests.get(self.hf_api_url, params=params)
            response.raise_for_status()
            
            models = response.json()
            if not models:
                raise ValueError(f"No models found for task: {task_type}")
            
            # Return the most downloaded model ID
            return models[0]["id"]
            
        except Exception as e:
            logger.error(f"Error searching Hugging Face API: {str(e)}")
            # Fallback to a default model
            return "bert-base-uncased"

    def download_model(self, model_id: str, task_type: str) -> Dict[str, any]:
        """Download and cache the model from Hugging Face"""
        if model_id not in self.loaded_models:
            try:
                logger.info(f"Downloading model: {model_id}")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id)
                self.loaded_models[model_id] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "pipeline": pipeline(task=task_type, 
                                      model=model, 
                                      tokenizer=tokenizer)
                }
            except Exception as e:
                logger.error(f"Error downloading model {model_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")
        return self.loaded_models[model_id]

model_agent = ModelAgent()

@app.post("/select-model/")
async def select_model(prompt: UserPrompt):
    """Endpoint to analyze prompt and dynamically select/load model from Hugging Face"""
    try:
        # Analyze the prompt
        task_type = model_agent.analyze_prompt(prompt.prompt)
        
        # Search Hugging Face for appropriate model
        model_id = model_agent.search_huggingface_models(task_type)
        
        # Download or get cached model
        # model_components = model_agent.download_model(model_id, task_type)
        
        return {
            "status": "success",
            "task_type": task_type,
            "model_id": model_id,
            "message": f"Model {model_id} selected and loaded for {task_type}"
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(model_agent.loaded_models)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)