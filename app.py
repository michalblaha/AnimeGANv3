from flask import Flask, request, jsonify, send_file

import os
import tempfile
import shutil
import subprocess

app = Flask(__name__)


@app.route('/models', methods=['GET'])
def list_onnx_files():
    directory = '/app/deploy'
    files = [file for file in os.listdir(directory) if file.endswith('.onnx')]
    files = [f.replace('/app/deploy/','').replace('AnimeGANv3_', '').replace('.onnx','') for f in files]
    return jsonify(files)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    model = request.args.get('model', default='AnimeGANv3_Hayao_36')  # Replace 'default_model' with your default
    device = request.args.get('device', default='cpu')  # cpu or gpu

    if file:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)

        # Assume the script modifies the file or creates a new one.
        # Update 'processed_file_path' to the path of the binary file you want to return.
        base, extension = os.path.splitext(temp_file_path)
        processed_file_path = base + ".done" + extension

        try:
            subprocess.run(["python", "deploy/convert.py", "-i " + temp_file_path.strip(), "-m " + model, "-d " + device.strip()], check=True)
        except subprocess.CalledProcessError as e:
            shutil.rmtree(temp_dir)
            return jsonify({"error": "Error running script", "details": str(e)}), 500

        # Return the processed binary file
        return_data = send_file(processed_file_path, as_attachment=True)
        
        # Clean up
        shutil.rmtree(temp_dir)

        return return_data

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
