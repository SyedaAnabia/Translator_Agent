import os
from agents import OpenAIChatCompletionsModel, AsyncOpenAI, Agent, Runner, set_tracing_disabled, RunConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
set_tracing_disabled(True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize the OpenAI client
client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1/"
)

# Initialize the agent with the correct model and pass client as 'openai_client'
Translator = Agent(
    model=OpenAIChatCompletionsModel(
        model="meta-llama/llama-3.3-70b-instruct:free",
        openai_client=client
    ),
    name="Translator ",
    instructions="""  You are a translator AI. Your task is to translate text between English and Urdu only.
    If the input is in English, translate it to Urdu.
    If the input is in Urdu, translate it to English.
    Only return the translated sentence. Do not add any extra explanation.
    If the input is not in English or Urdu, respond with 'I can only translate between English and Urdu.""",
)


# Define the configuration for the agent

config=RunConfig(
     model=OpenAIChatCompletionsModel(
        model="meta-llama/llama-3.3-70b-instruct:free",  # Ensure this model is available in your OpenRouter account
        openai_client=client),
    model_provider="client",
    tracing_disabled=True,
)

# Run the agent
result=Runner.run_sync(
    Translator,
    input= input("Enter text to translate (English or Urdu): "),
    run_config=config

)
print( result.final_output)
