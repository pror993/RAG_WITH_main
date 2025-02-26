from google import genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
import json

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the input model
class Notes(BaseModel):
    text: str

# Load extraction categories from config file
with open('extraction_config.json', 'r') as config_file:
    config = json.load(config_file)
    extraction_categories = config['extraction_categories']

# Dynamically create the Details model based on extraction categories
details_fields = {category.replace(" ", "_"): (str, ...) for category in extraction_categories}
Details = create_model('Details', **details_fields)

# Define the output schema
class OutputSchema(BaseModel):
    summary: str
    details: Details

# Initialize the Gemini client
client = genai.Client(api_key='AIzaSyDXo5O9YZJSt-QQh_a6qJD0Jq7QJADzCHE')  # Replace with your actual API key

@app.post("/process_notes")
async def process_notes(notes: Notes):
    category_prompt = "\n".join([f"- {category}" for category in extraction_categories])
    
    prompt = f"""
    You are a BPO assistant. Summarize the following notes and extract key details for these categories:
    {category_prompt}

    The output must be valid JSON and strictly follow this format:
    {{
        "summary": "A brief summary of the conversation",
        "details": {{
            {", ".join([f'"{category.replace(" ", "_")}": "Extracted information for {category}"' for category in extraction_categories])}
        }}
    }}

    Ensure all keys and values are properly quoted. Do not include any text outside of this JSON structure.

    Notes: {notes.text}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt],
            config={
                'response_mime_type': 'application/json',
                'response_schema': OutputSchema,
            },
        )
        
        # Use instantiated objects
        output: OutputSchema = response.parsed
        
        return output.model_dump()
    except json.JSONDecodeError as e:
        return {"error": f"Failed to generate valid JSON: {str(e)}", "raw_response": response.text if 'response' in locals() else "No response generated"}
    except ValueError as e:
        return {"error": str(e), "raw_response": response.text if 'response' in locals() else "No response generated"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@app.get("/extraction_categories")
async def get_extraction_categories():
    return {"categories": extraction_categories}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8505)
