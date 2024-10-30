from flask import Flask, render_template

app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# Explore page route
@app.route('/explore')
def explore():
    return render_template('explore.html')

if __name__ == '__main__':
    app.run(debug=True)
