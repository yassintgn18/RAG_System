from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin
from datetime import datetime
from flask import render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import docx
import PyPDF2
import json
import uuid
import traceback
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama  # Change: use llama-cpp instead of transformers

# Initialize Flask app
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///maintenance_management.db'
app.config['SECRET_KEY'] = 'mysecretkey'
db = SQLAlchemy(app)


UPLOAD_FOLDER = 'uploads/'
DATA_FOLDER = 'data/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx'}
embedder = SentenceTransformer('all-MiniLM-L6-v2')

embedding_dim = 384
index = None
metadata_store = []

INDEX_FILE = os.path.join(DATA_FOLDER, "faiss.index")
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.json")



# Add these near the top with other initializations
INDEX_TEMP_FILE = os.path.join(DATA_FOLDER, "faiss_temp.index")
METADATA_TEMP_FILE = os.path.join(DATA_FOLDER, "metadata_temp.json")
index_temp = None
metadata_temp_store = []


# Load or initialize the temp index
try:
    if os.path.exists(INDEX_TEMP_FILE) and os.path.exists(METADATA_TEMP_FILE):
        index_temp = faiss.read_index(INDEX_TEMP_FILE)
        with open(METADATA_TEMP_FILE, 'r') as f:
            metadata_temp_store = json.load(f)
        print(f"[INFO] Loaded existing temp index with {index_temp.ntotal} vectors")
    else:
        # Create new temp index if doesn't exist
        index_temp = faiss.IndexFlatL2(embedding_dim)
        with open(METADATA_TEMP_FILE, 'w') as f:
            json.dump([], f)
        print("[INFO] Created new temp index")
except Exception as e:
    print(f"[ERROR] Could not load/create temp FAISS data: {str(e)}")
    print(traceback.format_exc())
    exit("[FATAL] Failed to initialize temp index. Aborting.")

# Initialize the index only once at startup (without recreating)
try:
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, 'r') as f:
            metadata_store = json.load(f)
        print(f"[INFO] Loaded existing index with {index.ntotal} vectors")
    else:
        raise FileNotFoundError("FAISS index or metadata file not found.")
except Exception as e:
    print(f"[ERROR] Could not load existing FAISS data: {str(e)}")
    print(traceback.format_exc())
    exit("[FATAL] Application requires an existing FAISS index. Aborting.")


# Load Mistral model using llama-cpp
print("[INFO] Loading Mistral language model...")
model_path = r"C:\Users\E14\Desktop\module 1_ Document GenAI API\mistral-7b-v0.1.Q4_0.gguf"

try:
    # Initialize the Mistral model with llama-cpp
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
    print("[SUCCESS] Mistral model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load Mistral model: {str(e)}")
    print(traceback.format_exc())
    print("[WARNING] Application will run but may not generate responses")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_docx_to_text(filepath):
    doc = docx.Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs)

def convert_pdf_to_text(filepath):
    text = ""
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def split_text_into_chunks(text, max_tokens=500, overlap_tokens=100):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0

    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_len = len(tokens)

        if current_chunk_token_count + token_len <= max_tokens:
            current_chunk.append(sentence)
            current_chunk_token_count += token_len
        else:
            # Save current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Create new chunk with overlap
            overlap = []
            while current_chunk_token_count > 0 and overlap_tokens > 0:
                last_sentence = current_chunk.pop()
                token_len = len(tokenizer.encode(last_sentence, add_special_tokens=False))
                overlap.insert(0, last_sentence)
                current_chunk_token_count -= token_len
                overlap_tokens -= token_len

            current_chunk = overlap + [sentence]
            current_chunk_token_count = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def save_data():
    try:
        faiss.write_index(index, INDEX_FILE)
        faiss.write_index(index_temp, INDEX_TEMP_FILE)  # Add this
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata_store, f)
        with open(METADATA_TEMP_FILE, 'w') as f:  # Add this
            json.dump(metadata_temp_store, f)
    except Exception as e:
        print(f"[ERROR] Failed to save data: {str(e)}")
        print(traceback.format_exc())


def generate_response(prompt, max_length=300):
    try:
        # # Generate response using llama-cpp
        # response = llm(
        #     prompt, 
        #     max_tokens=max_length,
        #     stop=["</s>", "QUESTION:", "CONTEXT:"],
        #     temperature=0.7
        # )

        response = llm(
            prompt, 
            max_tokens=300,  # Reduce from 450 to limit verbosity
            temperature=0.6,  # Lower randomness (from 0.7)
            # top_p=0.8,
            # repetition_penalty=1.5
        )
        
        # Extract the generated text from the response
        generated_text = response["choices"][0]["text"]
        
        # Extract the answer and follow-up question
        answer, follow_up_question = extract_follow_up_question(generated_text.strip())
        
        return answer, follow_up_question
        
    except Exception as e:
        print(f"[ERROR] Failed to generate response: {str(e)}")
        print(traceback.format_exc())
        return "I couldn't generate a response due to an error.", "Would you like to try asking your question differently?"
@app.route('/befor_upload')
def home_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global index_temp, metadata_temp_store
    
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or format'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Save the uploaded file first
        file.save(filepath)
        
        # Process the file and convert it to text
        if filename.endswith('.docx'):
            text = convert_docx_to_text(filepath)
        else:
            text = convert_pdf_to_text(filepath)

        # Try to remove the file now that we've extracted the text
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"[WARNING] Could not remove file {filepath}: {str(e)}")
            # Continue processing even if file removal fails

        if not text.strip():
            return jsonify({'error': 'No text found.'}), 400

        # Split the text into chunks with overlap
        raw_chunks = split_text_into_chunks(text, max_tokens=250, overlap_tokens=50)
        chunks = [chunk for chunk in raw_chunks if len(chunk) >= 10]

        print(f'Number of chunks: {len(chunks)}')

        if not chunks:
            return jsonify({'error': 'No valid text chunks.'}), 400

        # Generate embeddings for the chunks
        embeddings = embedder.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        print(f'Number of embeddings: {len(embeddings)}')
        
        print("[INFO] Clearing previous temporary index...")
        index_temp = faiss.IndexFlatL2(embedding_dim)  # Recreate empty index
        metadata_temp_store = []  # Clear metadata
        # Add embeddings to the new index
        index_temp.add(embeddings)

        # Add metadata for each chunk
        timestamp = datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            metadata_temp_store.append({
                'id': str(uuid.uuid4()),
                'text': chunk,
                'source_file': filename,
                'chunk_number': i,
                'timestamp': timestamp
            })

        # Save the index and metadata
        save_data()

        print(f"Main index: {index.ntotal} vectors, Temp index: {index_temp.ntotal} vectors LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL THIS SI FOR DEBUG")

        return redirect(url_for('ask_question'))

    except Exception as e:
        print(traceback.format_exc())
        # Try to remove the file if it exists and an error occurred
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as remove_error:
                print(f"[WARNING] Could not remove file after error: {str(remove_error)}")
        
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/ask')
def ask_question():
    return render_template('ask_question.html')







@app.route('/ask_api', methods=['POST'])
def ask_question_api():
    try:
        data = request.get_json()
        question = data.get('question')

        if not question or not question.strip():
            return jsonify({'error': 'Question cannot be empty'}), 400

        # Check if index is empty
        if index.ntotal == 0:
            default_response = "Please upload a document first."
            default_follow_up = "Would you like to learn how to upload documents?"
            
            # Save to ChatQuery with empty answer
            if current_user.is_authenticated:
                chat_query = ChatQuery(
                    user_id=current_user.user_id,
                    question=question.strip(),
                    chatbot_answer=default_response
                )
                db.session.add(chat_query)
                db.session.commit()

            return jsonify({
                'results': [],
                'generated_answer': default_response,
                'follow_up_question': default_follow_up
            })

        # Embed the question and search for relevant chunks
        question_embedding = embedder.encode([question])[0]
        question_vector = np.array(question_embedding).astype('float32').reshape(1, -1)

        k = min(3, max(index.ntotal, index_temp.ntotal))
        all_results = []

        # Main index
        if index.ntotal > 0:
            distances, indices = index.search(question_vector, k)
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                similarity = float(1 / (1 + dist))
                if idx < len(metadata_store):
                    all_results.append({
                        'text': metadata_store[idx]['text'],
                        'source_file': metadata_store[idx].get('source_file', 'Unknown'),
                        'similarity': similarity,
                        'is_temp': False
                    })

        # Temp index
        if index_temp.ntotal > 0:
            distances_temp, indices_temp = index_temp.search(question_vector, k)
            for i in range(len(indices_temp[0])):
                idx = indices_temp[0][i]
                dist = distances_temp[0][i]
                similarity = float(1 / (1 + dist))
                if idx < len(metadata_temp_store):
                    all_results.append({
                        'text': metadata_temp_store[idx]['text'],
                        'source_file': metadata_temp_store[idx].get('source_file', 'Unknown'),
                        'similarity': similarity,
                        'is_temp': True
                    })

        # Sort and filter
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        similarity_threshold = 0.6
        top_results = [r for r in all_results if r['similarity'] >= similarity_threshold][:5]

        if not top_results and all_results:
            top_results = all_results[:2]

        retrieved_texts = [r['text'] for r in top_results]

        if not retrieved_texts:
            answer = "I don't have enough information to answer this question."
            follow_up_question = "Would you like to upload more documents related to your question?"
        else:
            prompt = construct_prompt(question, retrieved_texts)
            answer, follow_up_question = generate_response(prompt, max_length=300)

        # Save question and answer to DB
        if current_user.is_authenticated:
            try:
                # Save the question and answer to the database only if the user is logged in
                chat_query = ChatQuery(
                    user_id=current_user.user_id,
                    query_text=question.strip(),
                    response_text=answer,
                    created_at=datetime.utcnow()
                )
                db.session.add(chat_query)
                db.session.commit()
                print("Query saved successfully")

            except Exception as e:
                print(f"[ERROR] Error saving to database: {str(e)}")

        # Format results for frontend
        display_results = [{
            'text': r['text'],
            'source_file': r['source_file'],
            'similarity': r['similarity']
        } for r in top_results]

        return jsonify({
            'results': display_results,
            'generated_answer': answer,
            'follow_up_question': follow_up_question
        })

    except Exception as e:
        print(f"[ERROR] Exception in ask_question_api: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Error processing question: {str(e)}'}), 500


def construct_prompt(question, contexts):
    prompt_header = """  
You are a knowledgeable assistant that answers questions using only the provided CONTEXT.  

Role:  
- Provide factual, concise, and well-structured answers.  
- If an answer is not found in CONTEXT, respond exactly:  
  "The answer is not available in the provided information."  

Instructions:  
- Read and use ONLY the text under CONTEXT.  
- Answer fully, covering who, what, when, why, and how when relevant.  
- Structure responses using bullet points (one point per line) or numbered lists.  
- Do NOT hallucinate or add external information.
- ALWAYS end your response with a follow-up question prefixed by "Follow-up Question:" on its own line.
- Make the follow-up question engaging, thoughtful, and directly related to what the user asked.

Output Format:  
- Use `-` for bullets or `1.` `2.` for lists.  
- Each bullet/list item on its own line.  
- Keep the answer under 200 words unless more detail is necessary.
- Place "Follow-up Question:" on its own line after your answer.
  
Example:  
Question: What are neural networks?  
Answer:  
1. Neural networks are computational models inspired by the human brain's structure and function.
2. They consist of interconnected nodes (neurons) organized in layers.
3. Each connection has a weight that adjusts as learning proceeds.
4. They excel at pattern recognition, classification, and prediction tasks.
5. Deep neural networks contain multiple hidden layers, enabling them to learn complex representations.

Follow-up Question: Would you like to explore the different types of neural networks and their specific applications?
"""  

    formatted_contexts = "\n\n===\n\n".join(contexts)
    prompt = f"""{prompt_header}

CONTEXT:
{formatted_contexts}

QUESTION: {question}

ANSWER:"""

    max_prompt_tokens = 1500
    token_count = len(prompt.split()) * 1.3

    if token_count > max_prompt_tokens:
        truncated_contexts = []
        total_tokens = 0
        for context in contexts:
            context_token_count = len(context.split()) * 1.3
            if total_tokens + context_token_count + 100 < max_prompt_tokens - 100:
                truncated_contexts.append(context)
                total_tokens += context_token_count
            else:
                break

        if not truncated_contexts and contexts:
            first_context = contexts[0]
            words = first_context.split()
            max_words = int((max_prompt_tokens - 200) / 1.3)
            truncated_context = " ".join(words[:max_words])
            truncated_contexts = [truncated_context]

        formatted_contexts = "\n\n===\n\n".join(truncated_contexts)
        prompt = f"""{prompt_header}

CONTEXT:
{formatted_contexts}

QUESTION: {question}

ANSWER:"""

    return prompt



def extract_follow_up_question(response):
    """
    Extract the answer and follow-up question from the model's response.
    """
    follow_up_marker = "Follow-up Question:"
    
    if follow_up_marker in response:
        parts = response.split(follow_up_marker)
        answer = parts[0].strip()
        follow_up_question = parts[1].strip()
    else:
        # If no follow-up question is found, provide a default one
        answer = response.strip()
        follow_up_question = "there is no follow up question for this answer!!"
    
    return answer, follow_up_question

def clean_response(response):
    """
    Clean up the response from Mistral
    """
    # Remove any trailing or leading whitespace
    response = response.strip()
    
    # If the response has stopped mid-sentence, add an ellipsis
    if response and not response.endswith(('.', '!', '?')):
        response += "..."
    
    # If the response is too short or doesn't make sense
    if len(response.split()) < 5:
        response = "I couldn't generate a complete answer based on the available information."
    
    # Remove any remaining prompt artifacts
    for artifact in ["CONTEXT:", "QUESTION:", "ANSWER:"]:
        if response.startswith(artifact):
            response = response[len(artifact):].strip()
    
    return response

# if __name__ == '__main__':
#     app.run(debug=True)





# Flask-Login configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -----------------------------
# User Loader for Login Manager
# -----------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



# Routes
@app.route('/', methods=['GET', 'POST'])
def Test():
    return render_template("home.html")

@app.route('/amdin_page', methods=['GET', 'POST'])
def admin():
    return render_template("admin_page.html")

from werkzeug.security import check_password_hash
from flask_login import login_user

@app.route('/Login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")

            # ✅ Check if it's the admin
            if user.name == 'admin' and user.email == 'admin@gmail.com' and password == 'password123':
                return redirect(url_for('admin'))
            return redirect(url_for('Test'))  # Normal user redirection
        else:
            flash("Invalid email or password.", "error")

    return render_template("login.html")




@app.route('/SignUp', methods=['GET', 'POST'])
def SignUp():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role_id = int(request.form['role'])

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("SignUp.html")

        hashed_password = generate_password_hash(password)

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "error")
            return render_template("SignUp.html")

        new_user = User(name=name, email=email, password=hashed_password, role_id=role_id)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully!", "success")
        return redirect(url_for('login'))

    return render_template("SignUp.html")




@app.route('/upload_page')
def upload_documents():
    return render_template('uplaod_document_for_Admin.html')

@app.route('/upload_documents_for_admin', methods=['POST'])
def upload_documents_for_admin():
    file = request.files.get('document')

    if not file:
        print("the file has not been sent to the backend")
        return jsonify({'error': 'No file sent'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Save uploaded file
        file.save(filepath)

        # Convert the file to text
        if filename.endswith('.docx'):
            text = convert_docx_to_text(filepath)
        else:
            text = convert_pdf_to_text(filepath)

        # Delete file after reading
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"[WARNING] Could not remove file {filepath}: {str(e)}")

        if not text.strip():
            return jsonify({'error': 'No text found in document'}), 400

        # Split text into chunks
        raw_chunks = split_text_into_chunks(text, max_tokens=500, overlap_tokens=100)
        chunks = [chunk for chunk in raw_chunks if len(chunk) >= 10]

        if not chunks:
            return jsonify({'error': 'No valid text chunks.'}), 400

        print(f'[INFO] Number of valid chunks: {len(chunks)}')

        # Generate embeddings
        embeddings = embedder.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        print(f'[INFO] Number of embeddings: {len(embeddings)}')

        # Add to existing FAISS index
        index.add(embeddings)

        # Append to metadata
        timestamp = datetime.now().isoformat()
        for i, chunk in enumerate(chunks):
            metadata_store.append({
                'id': str(uuid.uuid4()),
                'text': chunk,
                'source_file': filename,
                'chunk_number': i,
                'timestamp': timestamp
            })

        # Save index and metadata to disk
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata_store, f)

        print("[INFO] New vectors and metadata added successfully")
        return redirect(url_for('upload_documents'))

    except Exception as e:
        print(traceback.format_exc())
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as remove_error:
                print(f"[WARNING] Could not remove file after error: {str(remove_error)}")

        return jsonify({'error': f'Processing error: {str(e)}'}), 500


from flask import flash, render_template, request, redirect, url_for
from werkzeug.security import generate_password_hash
from flask_login import login_required, current_user

# Updated manage_users route
@app.route('/manage_users')
@login_required
def manage_users():
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))
    
    users = User.query.all()
    roles = Role.query.all()
    return render_template('manage_users.html', users=users, roles=roles)

# Add user route
@app.route('/add_user', methods=['POST'])
@login_required
def add_user():
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))
    
    try:
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role_id = int(request.form['role_id'])
        
        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "danger")
            return redirect(url_for('manage_users'))
        
        # Hash password
        hashed_password = generate_password_hash(password)
        
        # Create new user
        new_user = User(name=name, email=email, password=hashed_password, role_id=role_id)
        db.session.add(new_user)
        db.session.commit()
        
        # Log the action
        new_log = Log(
            user_id=current_user.user_id,
            action=f"Created new user: {email}",
            details=f"Admin {current_user.email} created user {email} with role_id {role_id}"
        )
        db.session.add(new_log)
        db.session.commit()
        
        flash("User added successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error adding user: {str(e)}", "danger")
    
    return redirect(url_for('manage_users'))

# Update user route
@app.route('/update_user/<int:user_id>', methods=['POST'])
@login_required
def update_user(user_id):
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))
    
    try:
        user = User.query.get_or_404(user_id)
        
        # Update user data
        user.name = request.form['name']
        
        # Check if email is changed and if it already exists
        new_email = request.form['email']
        if new_email != user.email:
            existing_user = User.query.filter_by(email=new_email).first()
            if existing_user:
                flash("Email already registered.", "danger")
                return redirect(url_for('manage_users'))
            user.email = new_email
        
        # Update password if provided
        password = request.form['password']
        if password:
            user.password = generate_password_hash(password)
        
        # Update role
        user.role_id = int(request.form['role_id'])
        
        db.session.commit()
        
        # Log the action
        new_log = Log(
            user_id=current_user.user_id,
            action=f"Updated user: {user.email}",
            details=f"Admin {current_user.email} updated user {user.email} (ID: {user_id})"
        )
        db.session.add(new_log)
        db.session.commit()
        
        flash("User updated successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating user: {str(e)}", "danger")
    
    return redirect(url_for('manage_users'))

# Delete user route
@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))
    
    try:
        user = User.query.get_or_404(user_id)
        
        # Prevent admin from deleting themselves
        if user.email == 'admin@gmail.com':
            flash("Cannot delete admin account.", "danger")
            return redirect(url_for('manage_users'))
        
        user_email = user.email
        
        # Log the action before deletion
        new_log = Log(
            user_id=current_user.user_id,
            action=f"Deleted user: {user_email}",
            details=f"Admin {current_user.email} deleted user {user_email} (ID: {user_id})"
        )
        db.session.add(new_log)
        
        # Delete user
        db.session.delete(user)
        db.session.commit()
        
        flash("User deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting user: {str(e)}", "danger")
    
    return redirect(url_for('manage_users'))




# @app.route('/view_query_logs')
# def view_query_logs():
#     return "Query logs page coming soon"


# View past queries route
@app.route('/view_query_logs')
@login_required
def view_query_logs():
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))

    # Fetch past queries from the database
    queries = ChatQuery.query.all()
    
    # Return the page with queries
    return render_template('view_query_logs.html', queries=queries)

# Update response route
@app.route('/update_response/<int:query_id>', methods=['POST'])
@login_required
def update_response(query_id):
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))
    
    try:
        query = ChatQuery.query.get_or_404(query_id)
        
        # Update the response
        query.response_text = request.form['response_text']
        db.session.commit()

        # Log the action
        new_log = Log(
            user_id=current_user.user_id,
            action=f"Updated response for query ID: {query_id}",
            details=f"Admin {current_user.email} updated the response for query ID: {query_id}"
        )
        db.session.add(new_log)
        db.session.commit()

        flash("Response updated successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating response: {str(e)}", "danger")

    return redirect(url_for('view_query_logs'))

# Delete query route
@app.route('/delete_query/<int:query_id>', methods=['POST'])
@login_required
def delete_query(query_id):
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))

    try:
        query = ChatQuery.query.get_or_404(query_id)

        # Log the action before deletion
        new_log = Log(
            user_id=current_user.user_id,
            action=f"Deleted query ID: {query_id}",
            details=f"Admin {current_user.email} deleted query ID: {query_id}"
        )
        db.session.add(new_log)

        # Delete query
        db.session.delete(query)
        db.session.commit()

        flash("Query deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting query: {str(e)}", "danger")

    return redirect(url_for('view_query_logs'))


@app.route('/bulk_upload')
def bulk_upload():
    return render_template('bulk_upload.html')

@app.route('/process_bulk_upload', methods=['POST'])
def process_bulk_upload():
    files = request.files.getlist('documents')
    
    if not files or len(files) == 0 or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    results = {
        'success': 0,
        'failed': 0,
        'failed_files': []
    }
    
    for file in files:
        if not allowed_file(file.filename):
            results['failed'] += 1
            results['failed_files'].append(f"{file.filename} (invalid format)")
            continue
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            # Save uploaded file
            file.save(filepath)
            
            # Convert the file to text
            if filename.endswith('.docx'):
                text = convert_docx_to_text(filepath)
            else:
                text = convert_pdf_to_text(filepath)
                
            # Delete file after reading
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"[WARNING] Could not remove file {filepath}: {str(e)}")
                
            if not text.strip():
                results['failed'] += 1
                results['failed_files'].append(f"{filename} (no text extracted)")
                continue
                
            # Split text into chunks
            raw_chunks = split_text_into_chunks(text, max_tokens=500, overlap_tokens=100)
            chunks = [chunk for chunk in raw_chunks if len(chunk) >= 10]
            
            if not chunks:
                results['failed'] += 1
                results['failed_files'].append(f"{filename} (no valid chunks)")
                continue
                
            print(f'[INFO] {filename}: Number of valid chunks: {len(chunks)}')
            
            # Generate embeddings
            embeddings = embedder.encode(chunks, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            
            # Add to existing FAISS index
            index.add(embeddings)
            
            # Append to metadata
            timestamp = datetime.now().isoformat()
            for i, chunk in enumerate(chunks):
                metadata_store.append({
                    'id': str(uuid.uuid4()),
                    'text': chunk,
                    'source_file': filename,
                    'chunk_number': i,
                    'timestamp': timestamp
                })
                
            # Save index and metadata to disk
            faiss.write_index(index, INDEX_FILE)
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata_store, f)
                
            results['success'] += 1
            print(f"[INFO] Successfully processed {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {str(e)}")
            print(traceback.format_exc())
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as remove_error:
                    print(f"[WARNING] Could not remove file after error: {str(remove_error)}")
                    
            results['failed'] += 1
            results['failed_files'].append(f"{filename} (processing error)")
    
    # After processing all files, save the index and metadata once more to ensure everything is saved
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata_store, f)
    except Exception as e:
        print(f"[ERROR] Failed to save final index and metadata: {str(e)}")
        
    return jsonify({
        'success': True,
        'message': f"Successfully processed {results['success']} files, failed to process {results['failed']} files.",
        'details': results
    })

@app.route('/chatbot_feedback')
@login_required
def chatbot_feedback():
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))
    
    # Fetch past queries from the database
    queries = ChatQuery.query.all()
   
    # Return the page with queries
    return render_template('chatbot_feedback.html', queries=queries)

@app.route('/update_response_and_index/<int:query_id>', methods=['POST'])
@login_required
def update_response_and_index(query_id):
    # Check if user is admin
    if current_user.email != 'admin@gmail.com':
        flash("Access denied. Admin privileges required.", "danger")
        return redirect(url_for('Test'))
    
    try:
        # Get the query and its previous response
        query = ChatQuery.query.get_or_404(query_id)
        old_response = query.response_text
        new_response = request.form['response_text']
        
        # Only proceed if the response has actually changed
        if old_response != new_response:
            # Update the response in the database
            query.response_text = new_response
            db.session.commit()
            
            # Now add this Q&A pair to the vector database
            # Create a document that contains both the question and answer
            qa_text = f"Question: {query.query_text}\nAnswer: {new_response}"
            
            # Generate chunk for this Q&A pair
            chunks = [qa_text]  # Since this is already a small, focused piece of content
            
            # Generate embeddings
            embeddings = embedder.encode(chunks, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            
            # Add to existing FAISS index
            index.add(embeddings)
            
            # Add to metadata store
            timestamp = datetime.now().isoformat()
            metadata_store.append({
                'id': str(uuid.uuid4()),
                'text': qa_text,
                'source_file': 'admin_feedback',
                'chunk_number': 0,
                'timestamp': timestamp,
                'original_query_id': query_id
            })
            
            # Save index and metadata to disk
            faiss.write_index(index, INDEX_FILE)
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata_store, f)
            
            # Log the action
            new_log = Log(
                user_id=current_user.user_id,
                action=f"Updated response and indexed for query ID: {query_id}",
                details=f"Admin {current_user.email} updated the response for query ID: {query_id} and added to vector database"
            )
            db.session.add(new_log)
            db.session.commit()
            
            flash("Response updated and indexed successfully. The chatbot will use this answer for similar questions in the future.", "success")
        else:
            flash("No changes were made to the response.", "info")
            
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating response: {str(e)}", "danger")
    
    return redirect(url_for('chatbot_feedback'))





@app.route('/data')
def data_in_db():
    roles = Role.query.all()
    users = User.query.all()
    documents = Document.query.all()
    queries = ChatQuery.query.all()
    logs = Log.query.all()
    tasks = Task.query.all()
    
    print("admin saved succfuly")


    

    return render_template("data_in_db.html", roles=roles, users=users, documents=documents,
                            queries=queries, logs=logs, tasks=tasks)



@app.route('/index_info')
def index_info():
    try:
        # Load the FAISS index from disk
        if os.path.exists(INDEX_FILE):
            loaded_index = faiss.read_index(INDEX_FILE)
            num_vectors = loaded_index.ntotal
        else:
            num_vectors = 0

        # Return a simple HTML page
        return f"""
        <html>
            <head><title>Index Info</title></head>
            <body style="font-family: Arial; text-align: center; margin-top: 50px;">
                <h2>FAISS Index Info</h2>
                <p>Number of vectors stored in the index:</p>
                <h1 style="color: green;">{num_vectors}</h1>
            </body>
        </html>
        """
    except Exception as e:
        return f"<p>Error: {str(e)}</p>", 500


# -------------------
# Database Models
# -------------------

class Role(db.Model):
    __tablename__ = 'roles'
    role_id = db.Column(db.Integer, primary_key=True)
    role_name = db.Column(db.String(50), nullable=False)
    users = db.relationship('User', backref='role', lazy=True)


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.role_id'), nullable=False)

    def get_id(self):
        return str(self.user_id)

    documents = db.relationship('Document', backref='uploader', lazy=True)
    queries = db.relationship('ChatQuery', backref='user', lazy=True)
    logs = db.relationship('Log', backref='user', lazy=True)

    assigned_tasks = db.relationship('Task', backref='assigned_to_user', foreign_keys='Task.assigned_to', lazy=True)
    delegated_tasks = db.relationship('Task', backref='assigned_by_user', foreign_keys='Task.assigned_by', lazy=True)


class Document(db.Model):
    __tablename__ = 'documents'
    document_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    permanent = db.Column(db.Boolean, default=False)


class ChatQuery(db.Model):
    __tablename__ = 'chat_queries'
    query_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    query_text = db.Column(db.Text, nullable=False)
    response_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Log(db.Model):
    __tablename__ = 'logs'
    log_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text)


class Task(db.Model):
    __tablename__ = 'tasks'
    task_id = db.Column(db.Integer, primary_key=True)
    assigned_to = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    assigned_by = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    task_description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default='in progress')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)



with app.app_context():
    db.create_all()
    print("All tables created successfully!")


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        app.run(debug=True)



