import openai
import os
import backoff
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIStatusError, APIConnectionError

# Load environment variables from the .env file
# print("Before loading .env:", os.getenv('OPENAI_API_KEY'))
load_dotenv()
# print("After loading .env:", os.getenv('OPENAI_API_KEY'))

# Set OpenAI API key from environment variable
key = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key
client = OpenAI(api_key  = key)

# Retry only on transient failures. NOT bare APIStatusError -- that also covers
# 400 (bad request), 402 (out of credit) and 404, none of which get better by
# retrying. Provider pinning (allow_fallbacks=False) means OpenRouter cannot
# route around an upstream limit for us, so we absorb it here instead.
def _is_transient(exc):
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    code = getattr(exc, "status_code", None)
    return code in (408, 409, 425, 429, 500, 502, 503, 504, 529)


RETRYABLE = (RateLimitError, APIStatusError, APIConnectionError)


@backoff.on_exception(backoff.expo, RETRYABLE, max_time=300, max_tries=6,
                      giveup=lambda e: not _is_transient(e))  # Use RateLimitError directly
def get_completion_with_backoff(**kwargs):
    """Get completion from OpenAI API with backoff in case of RateLimitError."""
    return client.chat.completions.create(**kwargs)  # Corrected method for chat-based models like GPT-4

def get_response(prompt):
    """Function to interact with GPT-4o and generate a response."""
    
    # Define the messages in a variable called 'msgs'
    msgs = [
        {"role": "system", "content": "You are an intelligent assistant helping a human in a collaborative game to collect a gem the human desires. Your task is to interpret the instruction provided by the human and generate an appropriate response enabling the human retrieve their desired gem."
},
        {"role": "user", "content": prompt}
    ]
    
    Model = "gpt-4o"
    
    response = client.chat.completions.create(
        model=Model,
        messages=msgs,
        max_tokens=512,
        temperature=0.2,
        #top_p = 0.9
    )
    return response.choices[0].message.content.strip()

# Main function to run the script and test the API connection
if __name__ == '__main__':
    # Check if the API key is correctly loaded
    
    if key:
        print("API key loaded successfully.")
        #PRINT THE LIST OF models for this key
        print(key)
        models = client.models.list()
        print("Models available for this API key:")
        for model in models:
            print(model.id)
            
        # Test with a sample prompt
        test_prompt = "Are you using the gpt-4o model?"
        result = get_response(test_prompt)
        print("GPT-4o Response:", result)
    else:
        print("API key not found. Please set it in your .env file.")

