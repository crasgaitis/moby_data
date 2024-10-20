import math
import numpy as np
from flask import Flask, Response, render_template, request
from utils import generate_frames, generate_frames_edge

app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# CV page routes
@app.route('/cv')
def cv():
    return render_template('cv.html')

@app.route('/cv2')
def cv2():
    return render_template('cv2.html')

@app.route('/force')
def force():
    return render_template('force.html')

# paramterization route
@app.route('/parameterization')
def parameterization():
    return render_template('parameterization.html')

# Vid route
@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames("dim"), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_angle')
def angle_video_feed():
    return Response(generate_frames("angle"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/longest_edge_stream')
def longest_edge_stream():
    return Response(generate_frames_edge(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calculate_parameters', methods=['POST'])
def calculate_parameters():
    passive_force = float(request.form['passive_force'])
    active_force = float(request.form['active_force'])
    width = float(request.form['width'])
    thickness = float(request.form['thickness'])
    length = float(request.form['length'])
    angle = float(request.form['angle'])

    delta_l_calc = lambda F, a: ((-(-10) + math.sqrt((-10)**2 - 4 * 7 * (10 - F * a))) / (2 * 7), (-(-10) - math.sqrt((-10)**2 - 4 * 7 * (10 - F * a))) / (2 * 7))
    a = length / 2
    F = ((9.8 * passive_force * 0.001) + (9.8 * active_force * 0.001)) / 2

    delta_l = delta_l_calc(F, a)
    c = (-2 * a * np.cos(angle) + np.sqrt((2 * a * np.cos(angle))**2 - 4 * (1) * (a**2 - length**2))) / 2
    calculated_length = np.round(a + c - delta_l, 2)
    calculated_angle = np.round(np.arccos((calculated_length**2 - a**2 - c**2) / (2 * a * c)))

    return render_template('parameterization.html', calculated_length=calculated_length[0], calculated_angle=calculated_angle[0])


if __name__ == '__main__':
    app.run(debug=True)
