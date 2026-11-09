import time
import threading
import urllib.request
from pathlib import Path

import cv2

try:
    import pygame
except ImportError:
    pygame = None

try:
    import winsound
except ImportError:
    winsound = None


SLEEP_THRESHOLD = 5.0
DROWSY_THRESHOLD = 2.0
WINDOW_NAME = "EyeGuard AI"
ALARM_FILE = Path(__file__).resolve().parent / "alarm.wav"

RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_cascade_file(filename):
    project_dir = Path(__file__).resolve().parent
    cascade_dir = project_dir / "haarcascades"

    installed_dir = getattr(cv2.data, "haarcascades", None)
    if installed_dir:
        installed_file = Path(installed_dir) / filename
        if installed_file.exists():
            return installed_file

    local_file = cascade_dir / filename
    if local_file.exists():
        return local_file

    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        f"data/haarcascades/{filename}"
    )
    try:
        cascade_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading {filename} from OpenCV.")
        with urllib.request.urlopen(url, timeout=20) as response:
            local_file.write_bytes(response.read())
        return local_file
    except Exception as error:
        print(f"[WARNING] Could not get {filename}: {error}")
        return None


def load_cascades():
    face_file = get_cascade_file("haarcascade_frontalface_default.xml")
    eye_file = get_cascade_file("haarcascade_eye.xml")
    if face_file is None or eye_file is None:
        print("[WARNING] Haar cascade files are unavailable.")
        return None, None

    face_cascade = cv2.CascadeClassifier(str(face_file))
    eye_cascade = cv2.CascadeClassifier(str(eye_file))
    if face_cascade.empty() or eye_cascade.empty():
        print("[WARNING] Could not load the Haar cascade files.")
        return None, None

    print("[OK] Loaded face and eye cascades.")
    return face_cascade, eye_cascade


class Alarm:
    def __init__(self):
        self.playing = False
        self.mode = "none"
        self.stop_beeps = threading.Event()
        self.beep_thread = None

        if ALARM_FILE.exists() and pygame is not None:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=1)
                pygame.mixer.music.load(str(ALARM_FILE))
                pygame.mixer.music.set_volume(1.0)
                self.mode = "pygame"
                return
            except Exception as error:
                print(f"[WARNING] Could not start pygame audio: {error}")

        if winsound is not None:
            self.mode = "winsound" if ALARM_FILE.exists() else "beep"
        else:
            print("[WARNING] No sound system available; using visual alerts only.")

    def _beep_loop(self):
        while not self.stop_beeps.is_set():
            try:
                winsound.Beep(2000, 500)
            except RuntimeError:
                try:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                except Exception as error:
                    print(f"[WARNING] Could not play the beep: {error}")
                    break
            if self.stop_beeps.wait(0.2):
                break

    def play(self):
        if self.playing:
            return

        self.playing = True
        try:
            if self.mode == "pygame":
                pygame.mixer.music.play(-1)
            elif self.mode == "winsound":
                winsound.PlaySound(
                    str(ALARM_FILE),
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
                )
            elif self.mode == "beep":
                self.stop_beeps.clear()
                self.beep_thread = threading.Thread(
                    target=self._beep_loop, daemon=True
                )
                self.beep_thread.start()
        except Exception as error:
            print(f"[WARNING] Could not play the alarm: {error}")

    def stop(self):
        if not self.playing:
            return

        self.playing = False
        try:
            if self.mode == "pygame":
                pygame.mixer.music.stop()
            elif self.mode == "winsound":
                winsound.PlaySound(None, winsound.SND_PURGE)
            elif self.mode == "beep":
                self.stop_beeps.set()
                if self.beep_thread is not None:
                    self.beep_thread.join(timeout=1.0)
                    self.beep_thread = None
        except Exception as error:
            print(f"[WARNING] Could not stop the alarm: {error}")

    def close(self):
        self.stop()
        if self.mode == "pygame":
            pygame.mixer.quit()


def main():
    face_cascade, eye_cascade = load_cascades()
    if face_cascade is None or eye_cascade is None:
        return

    alarm = Alarm()
    camera = None
    # Try the default webcam, then a couple of other camera indexes.
    for camera_index in (0, 1, 2):
        test_camera = cv2.VideoCapture(camera_index)
        if test_camera.isOpened():
            camera = test_camera
            print(f"[OK] Camera connected on index {camera_index}.")
            break
        test_camera.release()

    if camera is None:
        print("[WARNING] No camera detected.")
        alarm.close()
        return

    eyes_closed_since = None
    frame_count = 0
    fps = 0.0
    fps_start = time.time()

    try:
        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.3, 5, minSize=(80, 80)
            )
            eyes_detected = False

            # Check for eyes in the upper part of each detected face.
            for x, y, width, height in faces:
                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), BLUE, 2
                )
                face_gray = gray[y:y + height, x:x + width]
                upper_face = face_gray[:int(height * 0.6), :]
                eyes = eye_cascade.detectMultiScale(
                    upper_face, 1.1, 10, minSize=(25, 25)
                )
                for eye_x, eye_y, eye_width, eye_height in eyes:
                    cv2.rectangle(
                        frame,
                        (x + eye_x, y + eye_y),
                        (x + eye_x + eye_width, y + eye_y + eye_height),
                        YELLOW,
                        2,
                    )
                if len(eyes) > 0:
                    eyes_detected = True

            # Seeing eyes again resets the closed-eye timer.
            if eyes_detected:
                eyes_closed_since = None
                status = "Awake"
                alarm.stop()
            else:
                if eyes_closed_since is None:
                    eyes_closed_since = time.time()
                closed_for = time.time() - eyes_closed_since
                if closed_for >= SLEEP_THRESHOLD:
                    status = "Sleeping"
                    # The alarm only plays after the sleep threshold is reached.
                    alarm.play()
                elif closed_for >= DROWSY_THRESHOLD:
                    status = "Drowsy"
                    alarm.stop()
                else:
                    status = "Eyes Closed"
                    alarm.stop()

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, f"Status: {status}", (10, 35), FONT, 1.0, WHITE, 2)
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 25),
                FONT, 0.6, WHITE, 1,
            )
            if status == "Sleeping":
                cv2.putText(
                    frame, "!!! WAKE UP !!!", (150, 240), FONT, 1.5, RED, 3
                )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) <= 0:
                break
    finally:
        try:
            alarm.close()
        finally:
            try:
                camera.release()
            finally:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
