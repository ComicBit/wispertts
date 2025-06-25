import os
import sys
import uuid
import threading
import subprocess
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file

app = Flask(__name__)

JOBS = {}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

class Job:
    def __init__(self, infile, opts):
        self.id = str(uuid.uuid4())
        self.infile = infile
        self.opts = opts
        self.outfile = os.path.join(RESULT_DIR, f"{self.id}.txt")
        self.log = []
        self.status = 'running'

    def run(self):
        cmd = [sys.executable, 'script.py', self.infile, self.outfile]
        if self.opts.get('mode'):
            cmd += ['--mode', self.opts['mode']]
        if self.opts.get('local_model'):
            cmd += ['--local-model', self.opts['local_model']]
        if self.opts.get('diarize'):
            cmd.append('--diarize')
        if self.opts.get('language'):
            cmd += ['--language', self.opts['language']]
        if self.opts.get('timestamps'):
            cmd.append('--timestamps')
        if self.opts.get('aggregate'):
            cmd.append('--aggregate')

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            self.log.append(line)
        proc.wait()
        if proc.returncode == 0:
            self.status = 'done'
        else:
            self.status = 'error'
        self.log.append(f"[END] exit code {proc.returncode}\n")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    f = request.files['audio']
    filename = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{f.filename}")
    f.save(filename)
    opts = {
        'language': request.form.get('language') or None,
        'diarize': bool(request.form.get('diarize')),
        'mode': request.form.get('mode'),
        'local_model': request.form.get('local_model'),
        'aggregate': bool(request.form.get('aggregate')),
        'timestamps': bool(request.form.get('timestamps')),
    }
    job = Job(filename, opts)
    JOBS[job.id] = job
    threading.Thread(target=job.run, daemon=True).start()
    return redirect(url_for('status', job_id=job.id))

@app.route('/status/<job_id>')
def status(job_id):
    return render_template('status.html', job_id=job_id)

@app.route('/logs/<job_id>')
def logs(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404
    data = {'status': job.status, 'log': ''.join(job.log)}
    if job.status == 'done':
        with open(job.outfile, 'r', encoding='utf-8') as fh:
            data['result'] = fh.read()
    return jsonify(data)

@app.route('/download/<job_id>')
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job.status != 'done':
        return 'Not ready', 404
    return send_file(job.outfile, as_attachment=True, download_name='transcript.txt')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
