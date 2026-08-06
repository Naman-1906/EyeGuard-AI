import cv2
import time
import numpy as np
import sys
import os
import shutil
import threading
import urllib.request
from pathlib import Path

# Optional audio support for the alarm.
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not found. Trying alternative sound methods...")

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# App settings and thresholds
class Config:
    SLEEP_THRESHOLD    = 5.0
    DROWSY_THRESHOLD   = 2.0
    
    FACE_SCALE_FACTOR  = 1.3
    FACE_MIN_NEIGHBORS = 5
    FACE_MIN_SIZE      = (80, 80)
    
    EYE_SCALE_FACTOR   = 1.1
    EYE_MIN_NEIGHBORS  = 10
    EYE_MIN_SIZE       = (25, 25)
    
    WINDOW_NAME        = "EyeGuard AI"
    FRAME_WIDTH        = 640
    FRAME_HEIGHT       = 480
    
    COLOR_GREEN        = (0, 255, 0)
    COLOR_RED          = (0, 0, 255)
    COLOR_ORANGE       = (0, 165, 255)
    COLOR_BLUE         = (255, 0, 0)
    COLOR_YELLOW       = (0, 255, 255)
    COLOR_WHITE        = (255, 255, 255)
    COLOR_BLACK        = (0, 0, 0)
    
    ALARM_FILE         = "alarm.wav"
    ALARM_VOLUME       = 1.0
    BEEP_FREQUENCY     = 2000
    BEEP_DURATION      = 500
    
    FONT               = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE   = 1.0
    FONT_SCALE_SMALL   = 0.6
    FONT_THICKNESS     = 2

# Handle alarm playback and fallback beep mode
class SoundManager:
    def __init__(self, alarm_file):
        self.alarm_file = alarm_file
        self.is_playing = False
        self.sound_system = None
        self._beep_thread = None
        self._beep_thread_stop = threading.Event()
        self._initialize_sound()
    
    def _initialize_sound(self):
        if os.path.exists(self.alarm_file):
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=4096)
                    pygame.mixer.music.load(self.alarm_file)
                    pygame.mixer.music.set_volume(Config.ALARM_VOLUME)
                    self.sound_system = "pygame"
                    print("✅ Sound system: pygame initialized")
                    return
                except Exception as e:
                    print(f"⚠️  pygame initialization failed: {e}")
            if WINSOUND_AVAILABLE:
                self.sound_system = "winsound"
                print("✅ Sound system: winsound initialized with alarm file")
                return
        else:
            print(f"⚠️  Alarm file '{self.alarm_file}' not found!")
            if WINSOUND_AVAILABLE:
                self.sound_system = "winsound_beep"
                print("✅ Sound system: winsound fallback beep enabled")
                return
        
        if WINSOUND_AVAILABLE:
            self.sound_system = "winsound"
            print("✅ Sound system: winsound initialized")
            return
        
        self.sound_system = "none"
        print("⚠️  No sound system available. Visual only.")
    
    def _beep_loop(self):
        # Keep beeping until the user is detected awake again.
        while not self._beep_thread_stop.is_set():
            try:
                winsound.Beep(Config.BEEP_FREQUENCY, Config.BEEP_DURATION)
            except RuntimeError:
                try:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                except Exception as e:
                    print(f"⚠️  Could not play fallback beep: {e}")
                    break
            if self._beep_thread_stop.wait(0.2):
                break
    
    def play_alarm(self):
        if self.is_playing:
            return
        self.is_playing = True
        self._beep_thread_stop.clear()
        try:
            if self.sound_system == "pygame":
                pygame.mixer.music.play(-1)
            elif self.sound_system == "winsound":
                winsound.PlaySound(self.alarm_file, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            elif self.sound_system == "winsound_beep":
                self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
                self._beep_thread.start()
        except Exception as e:
            print(f"⚠️  Could not play alarm: {e}")
    
    def stop_alarm(self):
        if not self.is_playing:
            return
        self.is_playing = False
        try:
            if self.sound_system == "pygame":
                pygame.mixer.music.stop()
            elif self.sound_system == "winsound":
                winsound.PlaySound(None, winsound.SND_PURGE)
            elif self.sound_system == "winsound_beep":
                self._beep_thread_stop.set()
                if self._beep_thread is not None:
                    self._beep_thread.join(timeout=1.0)
                    self._beep_thread = None
        except Exception as e:
            print(f"⚠️  Could not stop alarm: {e}")
    
    def cleanup(self):
        self.stop_alarm()
        if self.sound_system == "pygame" and PYGAME_AVAILABLE:
            pygame.mixer.quit()

# Load Haar cascades and detect face/eye states
class EyeDetector:
    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self._load_cascades()
        self.eyes_closed_start = None
        self.current_status = "Awake"
        self.closure_duration = 0.0
        self.alarm_triggered = False
    
    def _download_cascade(self, filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
        target_path = self.script_dir / filename
        try:
            print(f"ℹ️  Downloading {filename} from OpenCV GitHub...")
            with urllib.request.urlopen(url, timeout=20) as response:
                data = response.read()
            if not data:
                raise RuntimeError("download returned no data")
            with target_path.open("wb") as file_handle:
                file_handle.write(data)
            print(f"✅ Saved {filename} to {target_path}")
            return str(target_path)
        except Exception as e:
            print(f"⚠️  Failed to download {filename}: {e}")
            return None
    
    def _get_cascade_file(self, filename):
        haarcascades_dir = getattr(cv2.data, "haarcascades", None)
        system_file = str(Path(haarcascades_dir) / filename) if haarcascades_dir else None
        local_file = str(self.script_dir / filename)
        if system_file and os.path.exists(system_file):
            return system_file
        if os.path.exists(local_file):
            return local_file
        downloaded = self._download_cascade(filename)
        if downloaded and os.path.exists(downloaded):
            return downloaded
        return None
    
    def _load_cascades(self):
        # Try system-installed cascade first, then local files, then download if needed.
        face_xml = 'haarcascade_frontalface_default.xml'
        eye_xml = 'haarcascade_eye.xml'

        face_file = self._get_cascade_file(face_xml)
        eye_file = self._get_cascade_file(eye_xml)

        if not face_file or not eye_file:
            print("⚠️ Haar cascade XML files are unavailable; running without face/eye detection.")
            self.face_cascade = None
            self.eye_cascade = None
            return

        self.face_cascade = cv2.CascadeClassifier(face_file)
        self.eye_cascade = cv2.CascadeClassifier(eye_file)

        if self.face_cascade is None or self.eye_cascade is None or self.face_cascade.empty() or self.eye_cascade.empty():
            print("⚠️ Haar cascade classifiers could not be loaded; running without face/eye detection.")
            self.face_cascade = None
            self.eye_cascade = None
            return

        print(f"✅ Cascade classifiers loaded successfully from: {face_file}, {eye_file}")
    
    def detect_faces(self, gray):
        if self.face_cascade is None:
            return []
        return self.face_cascade.detectMultiScale(gray, Config.FACE_SCALE_FACTOR, Config.FACE_MIN_NEIGHBORS, minSize=Config.FACE_MIN_SIZE)
    
    def detect_eyes(self, face_gray):
        if self.eye_cascade is None:
            return []
        height = face_gray.shape[0]
        upper_face = face_gray[0:int(height * 0.6), :]
        return self.eye_cascade.detectMultiScale(upper_face, Config.EYE_SCALE_FACTOR, Config.EYE_MIN_NEIGHBORS, minSize=Config.EYE_MIN_SIZE)
    
    def update_status(self, eyes_detected):
        current_time = time.time()
        if eyes_detected:
            self.eyes_closed_start = None
            self.closure_duration = 0.0
            self.current_status = "Awake"
            self.alarm_triggered = False
        else:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            self.closure_duration = current_time - self.eyes_closed_start
            if self.closure_duration >= Config.SLEEP_THRESHOLD:
                self.current_status = "Sleeping"
                self.alarm_triggered = True
            elif self.closure_duration >= Config.DROWSY_THRESHOLD:
                self.current_status = "Drowsy"
                self.alarm_triggered = False
            else:
                self.current_status = "Eyes Closed"
        return self.current_status, self.closure_duration, self.alarm_triggered

# Draw the video overlay and track FPS
class DisplayManager:
    def __init__(self):
        self.frame_count = 0
        self.fps_timer = time.time()
        self.fps = 0
    
    def calculate_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = time.time()
        return self.fps

    def draw_face_box(self, frame, x, y, w, h):
        cv2.rectangle(frame, (x, y), (x + w, y + h), Config.COLOR_BLUE, 2)
    
    def draw_eye_boxes(self, frame, fx, fy, eyes):
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), Config.COLOR_YELLOW, 2)
    
    def draw_status_panel(self, frame, status, duration, alarm, fps):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Status: {status}", (10, 35), Config.FONT, 1.0, Config.COLOR_WHITE, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 25), Config.FONT, 0.6, Config.COLOR_WHITE, 1)
        if alarm:
            cv2.putText(frame, "!!! WAKE UP !!!", (150, 240), Config.FONT, 1.5, Config.COLOR_RED, 3)
        return frame

# Main app loop for camera capture, detection, and alerting
class EyeGuardAI:
    def __init__(self):
        print("Initializing EyeGuard AI...")
        self.sound_manager = SoundManager(Config.ALARM_FILE)
        self.eye_detector = EyeDetector()
        self.display_manager = DisplayManager()
        self.camera = None
        self.running = True

        for camera_index in (0, 1, 2):
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                self.camera = camera
                print(f"✅ Camera connected on index {camera_index}")
                break
            camera.release()

        if self.camera is None:
            print("⚠️  No camera detected. Starting in headless mode.")
            self.running = False
        
    def run(self):
        while self.running:
            if self.camera is None or not self.camera.isOpened():
                print("⚠️  Camera unavailable. Exiting.")
                break

            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.eye_detector.detect_faces(gray)
            eyes_detected = False
            for (x, y, w, h) in faces:
                self.display_manager.draw_face_box(frame, x, y, w, h)
                eyes = self.eye_detector.detect_eyes(gray[y:y+h, x:x+w])
                if len(eyes) >= 1:
                    eyes_detected = True
                    self.display_manager.draw_eye_boxes(frame, x, y, eyes)
            
            status, duration, alarm = self.eye_detector.update_status(eyes_detected)
            if alarm: self.sound_manager.play_alarm()
            else: self.sound_manager.stop_alarm()
            
            fps = self.display_manager.calculate_fps()
            frame = self.display_manager.draw_status_panel(frame, status, duration, alarm, fps)
            
            cv2.imshow(Config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False
            
        self.cleanup()

    def cleanup(self):
        self.sound_manager.cleanup()
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EyeGuardAI()
    app.run()
