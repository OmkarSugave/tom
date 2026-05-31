import os
import tom
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Load the file using tom's loader
            df = tom.loader.file(filepath)
            if df is None:
                return jsonify({'error': 'Failed to parse the dataset file. Unsupported or corrupted format.'}), 400
            
            # 2. Run the describe pipeline to compile the HTML report
            tom.reporter.describe(df)
            
            # Verify the report was created
            report_path = "./tom_report/report.html"
            if os.path.exists(report_path):
                return jsonify({'success': True, 'report_url': '/tom_report/report.html'})
            else:
                return jsonify({'error': 'Diagnostics compiled, but report HTML file was not found.'}), 500
        except Exception as e:
            return jsonify({'error': f'Internal diagnostics error: {str(e)}'}), 500
            
@app.route('/tom_report/<path:path>')
def serve_report_files(path):
    return send_from_directory('tom_report', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
