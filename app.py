from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for, flash, send_file, make_response
from flask_cors import CORS
from pyngrok import ngrok
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
from functools import wraps
import threading
import queue
import uuid
from processing import processing
import torch
from urllib.parse import urlparse
import numpy as np
from flask_compress import Compress
import shutil
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import timedelta

# Base directory constants
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = os.path.join(STATIC_DIR, 'uploads')
USERS_FILE = os.path.join(STATIC_DIR, 'users.json')

# Standard file names
STANDARD_IMAGE_NAMES = ['front.png', 'left.png', 'back.png', 'right.png']
THUMBNAIL_FILENAME = 'thumbnail.png'
POINT_CLOUD_FILENAME_DRC = 'front.drc'
POINT_CLOUD_FILENAME_XYZ = 'front.xyz'
POINT_CLOUD_FILENAME_BIN = 'front.bin'
DONE_MARKER_FILENAME = 'done.drc'

# Valid referrers for security checks
VALID_GALLERY_REFERRERS = ['/gallery']

# Server configuration
STATIC_NGROK_DOMAIN = "visually-tolerant-sparrow.ngrok-free.app"
SERVER_PORT = 5000
SERVER_HOST = '0.0.0.0'

# ZoeDepth configuration
ZOE_REPO = "isl-org/ZoeDepth"
ZOE_MODEL = "ZoeD_N"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize users file if it doesn't exist
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)

# Dictionary to store processing status for each scan
processing_status = {}
# Thread-safe queue for image processing tasks
image_task_queue = queue.Queue()

print(f"Using device: {DEVICE}")
zoe = torch.hub.load(ZOE_REPO, ZOE_MODEL, pretrained=True).to(DEVICE)

def image_processor():
    while True:
        try:
            task = image_task_queue.get()
            if task is None:
                break  # Allows graceful shutdown

            scan_path = task['scan_path']
            scan_id = task['scan_id']
            username = task['username']

            processing_status[scan_id] = "processing"
            socketio.emit('status_update', 
                          {'scan_id': scan_id, 'status': 'processing'},
                          room=username)
            
            for file in STANDARD_IMAGE_NAMES:
                file_path = os.path.join(scan_path, file)
                print(f"[Worker] Processing image: {file_path}")
                process_image(file_path)
                print(f"[Worker] Done: {file_path}")

            processing_status[scan_id] = "completed"
            socketio.emit('status_update', 
                          {'scan_id': scan_id, 'status': 'completed'},
                          room=username)
            
        except Exception as e:
            print(f"[Worker] Error processing image: {e}")
        finally:
            image_task_queue.task_done()

# Start worker thread
worker_thread = threading.Thread(target=image_processor, daemon=True)
worker_thread.start()

def process_image(path):
    processing(path, zoe)  # Call the processing function from processing.py

def make_thumbnail(image_path, thumbnail_path):
    from rembg import remove
    from PIL import Image
    # Create a thumbnail version of the masked image with transparency
    img = Image.open(image_path).convert("RGB")
    extracted_np = remove(img, post_process_mask=True, only_mask=True)
    black_mask = np.invert(extracted_np)
    imga = np.array(img.convert("RGBA"))
    imga[:, :, 3] = np.invert(black_mask)
    thumbnail = Image.fromarray(imga)
    thumbnail.thumbnail((512, 512))
    thumbnail.save(thumbnail_path)

app = Flask(__name__)
Compress(app)
CORS(app)
app.secret_key = '12345'
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO

# Helper functions for user management
def get_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page')
            # Save the URL the user was trying to access
            session['next'] = request.url
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_user_folder(username):
    return os.path.join(UPLOAD_DIR, username)

def get_scan_folder(username, scan_id):
    return os.path.join(get_user_folder(username), scan_id)

def get_file_in_scan(username, scan_id, filename):
    return os.path.join(get_scan_folder(username, scan_id), filename)

@app.route('/')
def index():
    return redirect(url_for('home'))

# Route to serve the HTML page
@app.route('/home')
def home():
    return render_template('home.html')

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        print("Registering user")
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        users = get_users()
        
        if username in users:
            flash('Username already exists')
            return render_template('register.html')
        
        users[username] = {
            'password': generate_password_hash(password),
            'email': email
        }
        
        save_users(users)
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form 
        
        users = get_users()
        
        # Validate username and password
        if username in users and check_password_hash(users[username]['password'], password):
            session['username'] = username  # Store username in session
            flash('Login successful!')

            # If remember me is checked, create a persistent cookie
            if remember:
                resp = make_response(redirect(url_for('home')))
                resp.set_cookie('username', username, max_age=timedelta(days=30))  # Store username for 30 days
                return resp

            next_page = session.pop('next', None)
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid username or password')

    print("Checking for remember me cookie")
    username_cookie = request.cookies.get('username')
    print(f"Cookie username: {username_cookie}")
    if username_cookie:
        session['username'] = username_cookie
        return redirect(url_for('home'))

    return render_template('login.html')


# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove username from session

    # If there's a 'username' cookie, delete it
    resp = make_response(redirect(url_for('home')))
    resp.delete_cookie('username')  # Remove the remember me cookie
    flash('You have been logged out')
    
    return resp

# Route to handle image upload
@app.route('/single_upload', methods=['POST'])
@login_required
def upload_image():
    if 'image' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    
    username = session['username']
    user_folder = get_user_folder(username)
    os.makedirs(user_folder, exist_ok=True)

    file_path = os.path.join(user_folder, file.filename)
    file.save(file_path)

    image_task_queue.put(file_path)

    return jsonify({'message': f'Image saved at {file_path}', 'filename': file.filename}), 200

# Route to handle multiple image uploads
@app.route('/multiple_upload', methods=['POST'])
@login_required
def upload_images():
    # Get the number of images
    image_count = int(request.form.get('imageCount', 0))
    
    if image_count == 0:
        return jsonify({'message': 'No images in the request'}), 400
    
    if image_count != 4:
        return jsonify({'message': 'Exactly 4 images are required'}), 400
    
    username = session['username']
    user_folder = get_user_folder(username)
    os.makedirs(user_folder, exist_ok=True)
    
    # Generate a unique UUID for the new subfolder
    scan_id = str(uuid.uuid4())
    scan_folder = get_scan_folder(username, scan_id)
    os.makedirs(scan_folder, exist_ok=True)
    
    saved_files = []

    # Set initial status for this scan
    processing_status[scan_id] = "queued"
    socketio.emit('status_update', 
                 {'scan_id': scan_id, 'status': 'queued'},
                 room=username)
    
    # Process each image
    for i in range(1, image_count + 1):
        image_key = f'image{i}'
        
        if image_key not in request.files:
            return jsonify({'message': f'Missing image {i}'}), 400
        
        file = request.files[image_key]
        if file.filename == '':
            return jsonify({'message': f'Empty filename for image {i}'}), 400
        
        # Use the standard name based on the image index
        new_filename = STANDARD_IMAGE_NAMES[i-1]
        file_path = os.path.join(scan_folder, new_filename)
        file.save(file_path)

        if new_filename == STANDARD_IMAGE_NAMES[0]:  # 'front.png'
            thumbnail_path = os.path.join(scan_folder, THUMBNAIL_FILENAME)
            make_thumbnail(file_path, thumbnail_path)

        saved_files.append({
            'original_filename': file.filename,
            'new_filename': new_filename,
            'path': file_path
        })
    # Add to processing queue with metadata
    image_task_queue.put({
        'scan_path': scan_folder,
        'scan_id': scan_id,
        'username': username
    })
        
    return jsonify({
        'message': 'Successfully uploaded and renamed all 4 images',
        'scan_id': scan_id,
        'files': saved_files
    }), 200

@app.route('/gallery')
@login_required
def gallery():        
    username = session['username']
    user_folder = get_user_folder(username)
    
    if not os.path.exists(user_folder):
        folders = []
    else:
        folders = []
        for folder in os.listdir(user_folder):
            folder_path = os.path.join(user_folder, folder)
            if os.path.isdir(folder_path):
                thumbnail_path = os.path.join(folder_path, THUMBNAIL_FILENAME)
                if os.path.exists(thumbnail_path):
                    folders.append(folder)
    
    # Order the folders by creation time
    folders.sort(key=lambda x: os.path.getctime(os.path.join(user_folder, x)), reverse=True)

    return render_template('gallery.html', folders=folders)

@socketio.on('connect')
@login_required
def handle_connect():
    username = session['username']
    join_room(username)
    print(f"{username} connected to the socket.")

    user_statuses = {}
    user_folder = get_user_folder(username)
    if os.path.exists(user_folder):
        for folder in os.listdir(user_folder):
            folder_path = os.path.join(user_folder, folder)
            done_file_path = os.path.join(folder_path, DONE_MARKER_FILENAME)
            if os.path.isdir(folder_path) and os.path.exists(done_file_path):
                user_statuses[folder] = 'completed'
            elif os.path.isdir(folder_path) and processing_status.get(folder) != None:
                user_statuses[folder] = processing_status[folder]

    emit('initial_statuses', user_statuses, room=username)

@socketio.on('disconnect')
@login_required
def handle_disconnect():
    username = session['username']
    leave_room(username)
    print(f"{username} disconnected from the socket.")

# New route to serve the status page
@app.route('/status')
@login_required
def status_page():
    return render_template('status.html')

# Route to serve thumbnail images
@app.route('/model_thumbnail/<folder_name>')
@login_required
def model_thumbnail(folder_name):
    username = session['username']
    user_folder = get_scan_folder(username, folder_name)
    return send_from_directory(user_folder, THUMBNAIL_FILENAME)

@app.route('/download_model_pointcloud/<folder_name>')
@login_required
def download_model_pointcloud(folder_name):
    username = session['username']
    user_folder = get_scan_folder(username, folder_name)
    file_path = os.path.join(user_folder, POINT_CLOUD_FILENAME_DRC)
    return send_file(
        file_path,
        as_attachment=True,
        download_name=POINT_CLOUD_FILENAME_DRC,
        mimetype='application/octet-stream'
    )

@app.route('/view_model_pointcloud_drc/<folder_name>')
@login_required
def view_model_pointcloud_drc(folder_name):
    username = session['username']
    user_folder = get_scan_folder(username, folder_name)
    file_path = os.path.join(user_folder, POINT_CLOUD_FILENAME_DRC)
    return send_file(
        file_path,
        as_attachment=False,  # DOESN'T TRIGGER DOWNLOAD
        mimetype='application/octet-stream'
    )

@app.route('/view_model_glb/<folder_name>')
@login_required
def view_model_glb(folder_name):
    username = session['username']
    user_folder = get_scan_folder(username, folder_name)
    file_path = os.path.join(user_folder, "front.glb")
    return send_file(
        file_path,
        as_attachment=False,  # DOESN'T TRIGGER DOWNLOAD
        mimetype='application/octet-stream'
    )

@app.route('/delete_model/<folder_name>', methods=['POST'])
@login_required
def delete_model(folder_name):
    username = session['username']
    user_folder = get_scan_folder(username, folder_name)

    if os.path.exists(user_folder):
        shutil.rmtree(user_folder)
        flash(f'Model {folder_name} deleted successfully!')
    else:
        flash(f'Model {folder_name} not found!')
    
    return redirect(url_for('gallery'))

# Protect all image and file routes from direct access
@app.before_request
def block_direct_file_access():
    # Protect model_thumbnail and model_pointcloud routes
    if request.path.startswith('/model_thumbnail/') or request.path.startswith('/model_pointcloud/'):
        # Get the referrer's path if it exists
        referrer = request.referrer
        if not referrer:
            return redirect(url_for('gallery'))  # No referrer, redirect
            
        # Extract path from referrer URL
        referrer_path = urlparse(referrer).path
        
        # Check against valid referrers
        if referrer_path not in VALID_GALLERY_REFERRERS:
            return redirect(url_for('gallery'))

@app.route('/uploads/<filename>')
def serve_image(filename):
    username = session['username']
    user_folder = get_user_folder(username)
    return send_from_directory(user_folder, filename)

@app.route('/create')
@login_required
def create():
    return render_template('create.html')

@app.route('/help')
@login_required
def help():
    return render_template('help.html')

@app.route('/viewer/<folder_name>')
@login_required
def viewer(folder_name):
    username = session['username']
    user_folder = get_scan_folder(username, folder_name)
    file_path = os.path.join(user_folder, "front.glb")
    if os.path.exists(file_path):
        return render_template('glb_viewer.html', folder_name=folder_name)
    else:
        return render_template('viewer.html', folder_name=folder_name)

if __name__ == '__main__':
    public_url = ngrok.connect(addr=SERVER_PORT, domain=STATIC_NGROK_DOMAIN)
    print(f"Public URL: {public_url}")
    app.run(host=SERVER_HOST, port=SERVER_PORT)