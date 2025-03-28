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

# Configure Redis for rate limiting storage
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"]
)


# Load the pre-trained medical model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Set pad token to eos token as mentioned in the error
tokenizer.pad_token = tokenizer.eos_token

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
    user_input_lower = user_input.lower()
    medical_context = ""
    
    # Add condition-specific context to guide the model
    if any(word in user_input_lower for word in ["headache", "migraine"]):
        medical_context = "Provide information about causes, symptoms, and treatment options for headaches. Mention rest, hydration, and when to seek medical attention."
    elif any(word in user_input_lower for word in ["fever", "temperature"]):
        medical_context = "Discuss fever management, including rest, hydration, medication options, and when to see a doctor."
    elif any(word in user_input_lower for word in ["cough", "cold", "sore throat"]):
        medical_context = "Provide information about managing cold symptoms, including rest, fluids, over-the-counter medications, and when to seek medical care."
    elif any(word in user_input_lower for word in ["stomach", "nausea", "vomit", "diarrhea"]):
        medical_context = "Discuss digestive issues, including possible causes, management strategies, and when to consult a doctor."
    elif any(word in user_input_lower for word in ["skin", "rash", "itch"]):
        medical_context = "Provide general information about skin conditions, possible causes, basic care, and when dermatological consultation is recommended."
    elif any(word in user_input_lower for word in ["anxiety", "stress", "mental"]):
        medical_context = "Discuss mental health concerns compassionately, suggest coping strategies, and encourage professional support when needed."
    else:
        medical_context = "Provide helpful general health information while acknowledging limitations as an AI and encouraging professional medical consultation when appropriate."
    
    # Enhanced prompt with medical context
    prompt = f"Below is a medical question and the response from a qualified medical AI assistant.\n\nCONTEXT: {medical_context}\n\nUser: {user_input}\n\nMedical Assistant: "
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)

    try:
        # Generate response with adjusted parameters
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.5,  # Lower temperature for more focused responses
            do_sample=True,
            top_k=30,
            top_p=0.85,
            no_repeat_ngram_size=3,
            repetition_penalty=1.8
        )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Extract only the assistant's response part
        parts = full_response.split("Medical Assistant: ")
        if len(parts) > 1:
            response = parts[1].strip()
        else:
            response = ""
        
        logging.info(f"Generated response: {response}")
        
        # If response seems nonsensical or empty, return a default message
        if not response or len(response) < 20 or "???" in response or "!!!" in response:
            default_responses = {
                "headache": "Headaches can be caused by various factors including stress, dehydration, lack of sleep, or eye strain. For persistent or severe headaches, please consult a doctor. Rest, staying hydrated, and over-the-counter pain relievers may help with mild headaches.",
                "fever": "Fever is often a sign that your body is fighting an infection. Rest, stay hydrated, and take acetaminophen or ibuprofen to reduce fever. Seek medical attention for high fevers or fevers lasting more than three days.",
                "cold": "For coughs and colds, rest and staying hydrated are important. Over-the-counter medications may help relieve symptoms. If symptoms persist beyond 7-10 days or worsen significantly, please consult a healthcare provider.",
                "stress": "Managing stress through relaxation techniques, regular exercise, adequate sleep, and potentially speaking with a mental health professional can be beneficial for your overall health."
            }
            
            for key, value in default_responses.items():
                if key in user_input_lower:
                    response = value
                    break
            else:
                response = "I'm sorry, but I need to remind you that I'm a basic AI assistant without medical training. For any health concerns, please consult with a qualified healthcare professional."
        
        # Add medical disclaimer
        response += "\n\n*Note: This is AI-generated information and should not replace professional medical advice.*"
        
        return response
    
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, I couldn't generate a response. Please try again."

if __name__ == '__main__':
    app.run(debug=True, ssl_context='adhoc')  # Use HTTPS for secure communication
