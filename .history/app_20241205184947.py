from flask import Flask, request, jsonify, render_template
import markdown
from transformers import AutoTokenizer, AutoModelForCausalLM
import bleach
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Load the pre-trained medical model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForCausalLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
@limiter.limit("5 per minute")  # Rate limit for this endpoint
def ask():
    user_input = request.form['user_input']
    if not user_input:
        return jsonify({'error': 'Empty input'}), 400

    if len(user_input) > 512:
        return jsonify({'error': 'Input too long'}), 400

    try:
        # Sanitize user input
        sanitized_input = bleach.clean(user_input)
        response = generate_response(sanitized_input)
        formatted_response = markdown.markdown(response)
        return jsonify({'response': formatted_response})
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing your request. Please try again later.'}), 500

def generate_response(user_input):
    # Add context to the prompt to focus on medical topics
    prompt = f"Medical Assistant: {user_input}\nPlease provide a medically accurate response."
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)

    try:
        # Generate response with adjusted parameters
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,  # Adjusted temperature for more focused responses
            do_sample=True,
            top_k=50,   # Top-k sampling
            top_p=0.9,  # Top-p (nucleus) sampling
            no_repeat_ngram_size=2,  # Prevent repeating phrases
            repetition_penalty=2.0  # Penalize repeated tokens
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Remove the question from the response if it appears
        if response.startswith(user_input):
            response = response[len(user_input):].strip()

        # Log the generated response
        logging.info(f"Generated response: {response}")

        # Validate the response to ensure it is coherent and medically relevant
        if not response or any(char in response for char in ['[,!?]', '*', '(', ')', '{', '}', '\\', '&', '|', '_', '~', '@', '$', '#', '%', ';']):
            logging.error(f"Incoherent response generated: {response}")
            return "I'm sorry, I couldn't generate a coherent response. Please try again."

        return response

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate a response. Please try again."

if __name__ == '__main__':
    app.run(debug=True, ssl_context='adhoc')  # Use HTTPS for secure communication
