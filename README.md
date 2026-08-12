# Face-Recognition-Attendance-system
Real-time Face Recognition Attendance System using Python, Django, OpenCV and SQLite. Marks attendance automatically and prevents duplicates.
views.py
import os
import numpy as np
import cv2
import time
import pandas as pd
from datetime import datetime
import sqlite3
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.conf import settings


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(BASE_DIR, 'media', 'faces')
ATTENDANCE_DB = os.path.join(BASE_DIR, 'media','attendance.db')
CASCADE_PATH = os.path.join(BASE_DIR, 'attendance', 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


if face_cascade.empty():
    print("ERROR: HAAR CASCADE NOT LOADED. CHECK PATH:", CASCADE_PATH)


KNOWN_NAMES =['Anushka']
scan_line_y = 50
scan_direction = 1
attendance_marked = False
marked_name = ""
stop_timer = 0
scan_start_time = time.time()
session_start_time = time.time()

known_faces = {}

def load_known_faces():
    global known_faces
    known_faces = {}
    print("Loading faces...")
    for name in KNOWN_NAMES:
        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        print(f"Trying to load:{img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (100, 100))
            known_faces[name] = img
            print(f"Loaded: {name}")
        else:
            print(f"ERROR: {img_path} not found")

def recognize_face(face_img):
    global known_faces
    if not known_faces: 
        return "No Training Data", 0
    
    resized_face = cv2.resize(face_img, (100, 100))
    min_diff = float('inf')
    name = "Unknown"
    
    for known_name, known_face in known_faces.items():
        diff = np.linalg.norm(resized_face - known_face)
        print(f"Checking {known_name}, Difference: {diff:.2f}")
        
        if diff < min_diff:
            min_diff = diff
            name = known_name
    
    if min_diff < 15000:
        return name, min_diff
    return "Unknown", min_diff
load_known_faces()

def mark_attendance(name):
    print(f"DEBUG: mark_attendance called for {name}")
    
    if name == "Unknown" or name == "No Training Data":
        return False
    
    
    conn = sqlite3.connect(ATTENDANCE_DB)
    cursor = conn.cursor()
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Date TEXT NOT NULL,
            Time TEXT NOT NULL
        )
    ''')
    
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    print(f"DEBUG: Checking if {name} already marked on {date}")
    
    
    cursor.execute("SELECT * FROM attendance WHERE Name=? AND Date=?", (name, date))
    result = cursor.fetchone()
    
    if result is None: 
        cursor.execute("INSERT INTO attendance (Name, Date, Time) VALUES (?, ?, ?)", (name, date, time_str))
        conn.commit()
        print(f"Attendance marked for {name} in attendance.db")
        conn.close()
        return True
    
    print(f"DEBUG: {name} already marked today")
    conn.close()
    return False

def home(request):
    global scan_start_time, attendance_marked, stop_timer, marked_name
    scan_start_time = time.time()
    attendance_marked = False
    stop_timer = 0
    marked_name = ""
    return render(request, 'attendance/register.html')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def draw_corner_neon(img, x, y, w, h,blue, thickness=2):
    corner_len = 30
    cv2.line(img, (x, y), (x + corner_len, y),blue, thickness)
    cv2.line(img, (x, y), (x, y + corner_len),blue, thickness)
    cv2.line(img, (x + w - corner_len, y), (x + w, y),blue, thickness)
    cv2.line(img, (x + w, y), (x + w, y + corner_len),blue, thickness)
    cv2.line(img, (x, y + h - corner_len), (x, y + h),blue, thickness)
    cv2.line(img, (x, y + h), (x + corner_len, y + h),blue, thickness)
    cv2.line(img, (x + w - corner_len, y + h), (x + w, y + h),blue, thickness)
    cv2.line(img, (x + w, y + h - corner_len), (x + w, y + h),blue, thickness)
    return img

def draw_scanner_line_in_box(img, x, y, w, h, scan_y, color):
    thickness = 2
    if scan_y >= y and scan_y <= y+h:
        cv2.line(img, (x, scan_y), (x + w, scan_y), color, thickness)
        cv2.line(img, (x, scan_y), (x + w, scan_y), (255,255,255), 1)
    return img

def gen_frames():
    global scan_start_time, scan_line_y, scan_direction, attendance_marked, marked_name, stop_timer,session_start_time
    session_start_time = time.time()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ERROR: CAMERA NOT OPENED. TRY cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)")
        return

    while True:
        elapsed_time = time.time() - session_start_time
        if elapsed_time > 20:
            ret, frame = cap.read()
            cv2.putText(frame, "20 SECONDS OVER - CAMERA STOPPED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            break

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
            
        h, w, _ = frame.shape

        remaining = int(20 - elapsed_time)
        cv2.putText(frame, f"Time Left: {remaining}s", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        
        scan_start = 50
        scan_end = h - 20
        scan_line_y += 5 * scan_direction
        if scan_line_y >= scan_end or scan_line_y <= scan_start: 
            scan_direction *= -1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_detected = False
        if len(faces) == 0:
            cv2.putText(frame, "No Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        for (x,y,w,h) in faces:
            face_detected = True
            face_img = gray[y:y+h, x:x+w]
            name, confidence = recognize_face(face_img)
            
            if name!= "Unknown" and name!= "No Training Data":
                frame = draw_corner_neon(frame, x, y, w, h, (0,255,0))
                frame = draw_scanner_line_in_box(frame, x, y, w, h, scan_line_y, (0,255,0))
                cv2.putText(frame, f"{name}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                
                if not attendance_marked:
                    if mark_attendance(name):
                        attendance_marked = True
                        marked_name = name
                        stop_timer = 20
            else:
                frame = draw_corner_neon(frame, x, y, w, h, (0,0,255))
                frame = draw_scanner_line_in_box(frame, x, y, w, h, scan_line_y, (0,0,255))
                cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if stop_timer > 0:
            stop_timer -= 1
            cv2.putText(frame, f"ATTENDANCE MARKED FOR {marked_name}", (int(w/2)-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(frame, f"Closing in {stop_timer}s", (int(w/2)-100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if stop_timer == 0:
                break
        elif not face_detected:
            cv2.putText(frame, "Scanning... Please wait, Recognising Face", (int(w/2)-250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if frame is None or frame.size == 0:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
