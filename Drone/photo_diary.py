"""
fly_tello.py's photo diary: saves what the drone actually SAW during a
flight, next to its flight log, so a missed banana can be looked at and
the banana AI re-run on the real pictures afterwards.

    flight_logs/flight_<time>.csv           numbers, one row per picture
    flight_logs/flight_<time>_photos/       pictures:
        t0012.40_SCAN.jpg                   one per second
        t0015.85_APPROACH_banana.jpg        up to 4/s while a banana is seen

Pictures are the clean camera view (no boxes or text drawn on them) at
full Tello resolution. Saving happens on a background thread, so it
doesn't slow down the camera + fly brain loop (which needs ~15+
pictures/s to catch a hand). Like the flight logs, never uploaded to
GitHub (see .gitignore).
"""

import os
import queue
import threading

import cv2


# One picture this often...
EVERY_S = 1.0

# ...or this often while the banana AI sees a banana
BANANA_EVERY_S = 0.25

JPEG_QUALITY = 85

# Pictures waiting to be written; if the disk falls behind, extras are
# skipped rather than slowing the flight loop down
MAX_WAITING = 16


class PhotoDiary:

    def __init__(self, log_path):

        self.folder = os.path.splitext(log_path)[0] + "_photos"

        os.makedirs(self.folder, exist_ok=True)

        self._waiting = queue.Queue(maxsize=MAX_WAITING)

        self._last_t = None

        self.saved = 0

        self.skipped = 0

        self._thread = threading.Thread(target=self._write, daemon=True)

        self._thread.start()


    def maybe_save(self, frame_bgr, t, state, bananas_seen):
        """Call once per picture, BEFORE anything is drawn on it."""

        every = BANANA_EVERY_S if bananas_seen else EVERY_S

        if self._last_t is not None and t - self._last_t < every:
            return

        self._last_t = t

        name = f"t{t:07.2f}_{state}" + ("_banana" if bananas_seen else "") + ".jpg"

        try:

            self._waiting.put_nowait((os.path.join(self.folder, name), frame_bgr.copy()))

        except queue.Full:

            self.skipped += 1


    def _write(self):

        while True:

            item = self._waiting.get()

            if item is None:
                return

            path, frame_bgr = item

            if cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                self.saved += 1


    def close(self):
        """Finish writing what's waiting, then report where it went."""

        self._waiting.put(None)

        self._thread.join(timeout=5)

        print(f"Photo diary: {self.saved} pictures in {self.folder}"
              + (f" ({self.skipped} skipped - disk too slow)" if self.skipped else ""))
