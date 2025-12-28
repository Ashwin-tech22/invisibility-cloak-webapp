from flask import Flask, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def showcase():
    with open('showcase.html', 'r', encoding='utf-8') as f:
        content = f.read()
    return content

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)