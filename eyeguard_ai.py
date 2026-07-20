import cv2
import time
import numpy as np
import sys
import os
import threading
import urllib.request

# Try to import pygame for alarm sound
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not found. Trying alternative sound methods...")

# Try winsound for Windows users
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# ─── CONFIGURATION SETTINGS ──────────────────────────────────
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
    ALARM_VOLUME       = 0.8
    
    FONT               = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE   = 1.0
    FONT_SCALE_SMALL   = 0.6
    FONT_THICKNESS     = 2

# ─── SOUND MANAGER ───────────────────────────────────────────
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
        while not self._beep_thread_stop.wait(1.0):
            try:
                winsound.MessageBeep(winsound.MB_ICONHAND)
            except Exception as e:
                print(f"⚠️  Could not play fallback beep: {e}")
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

# ─── DETECTOR ────────────────────────────────────────────────
class EyeDetector:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self._load_cascades()
        self.eyes_closed_start = None
        self.current_status = "Awake"
        self.closure_duration = 0.0
        self.alarm_triggered = False
    
    def _download_cascade(self, filename):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
        target_path = os.path.join(self.script_dir, filename)
        try:
            print(f"ℹ️  Downloading {filename} from OpenCV GitHub...")
            urllib.request.urlretrieve(url, target_path)
            print(f"✅ Downloaded {filename} to {target_path}")
            return target_path
        except Exception as e:
            print(f"⚠️  Failed to download {filename}: {e}")
            return None
    
    def _get_cascade_file(self, filename):
        system_file = os.path.join(cv2.data.haarcascades, filename)
        local_file = os.path.join(self.script_dir, filename)
        if os.path.exists(system_file):
            return system_file
        if os.path.exists(local_file):
            return local_file
        downloaded = self._download_cascade(filename)
        if downloaded and os.path.exists(downloaded):
            return downloaded
        return None
    
    def _load_cascades(self):
        """Robustly load Haar Cascade XML files."""
        face_xml = 'haarcascade_frontalface_default.xml'
        eye_xml = 'haarcascade_eye.xml'

        face_file = self._get_cascade_file(face_xml)
        eye_file = self._get_cascade_file(eye_xml)

        if not face_file or not eye_file:
            raise IOError("❌ Could not locate cascade XML files. Ensure OpenCV haarcascade data or local XMLs are available.")

        self.face_cascade = cv2.CascadeClassifier(face_file)
        self.eye_cascade = cv2.CascadeClassifier(eye_file)

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise IOError("❌ Could not load cascade classifiers! Ensure XML files are present and valid.")

        print(f"✅ Cascade classifiers loaded successfully from: {face_file}, {eye_file}")
    
    def detect_faces(self, gray):
        return self.face_cascade.detectMultiScale(gray, Config.FACE_SCALE_FACTOR, Config.FACE_MIN_NEIGHBORS, minSize=Config.FACE_MIN_SIZE)
    
    def detect_eyes(self, face_gray):
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

# ─── DISPLAY MANAGER ─────────────────────────────────────────
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

# ─── MAIN APPLICATION ─────────────────────────────────────────
class EyeGuardAI:
    def __init__(self):
        print("Initializing EyeGuard AI...")
        self.sound_manager = SoundManager(Config.ALARM_FILE)
        self.eye_detector = EyeDetector()
        self.display_manager = DisplayManager()
        self.camera = cv2.VideoCapture(0)
        self.running = True
        
    def run(self):
        while self.running:
            ret, frame = self.camera.read()
            if not ret: continue
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
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EyeGuardAI()
    app.run()