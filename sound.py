import threading
import time
import os

def beep_alarm_paplay(sound_file, repeat=3, interval=0.5, stop_event=None):
    """
    Sounds a beeping alarm using paplay on Linux in a non-blocking way.

    Args:
        sound_file: Path to the sound file (e.g., "beep.ogg").
        repeat: Number of times to repeat the beep.
        interval: Time interval between beeps in seconds.
        stop_event: threading.Event to allow stopping the alarm. If None, the alarm runs to completion.
    """
    def _beep_thread():
        for _ in range(repeat):
            if stop_event and stop_event.is_set():
                break
            try:
                os.system(f"paplay '{sound_file}'")  # Use paplay to play the sound
            except Exception as e:
                print(f"Error playing sound: {e}")
                break # if paplay fails, stop the sequence.

            if _ < repeat - 1:
                if stop_event:
                    start = time.time()
                    while time.time() - start < interval:
                        if stop_event.is_set():
                            return
                else:
                    time.sleep(interval)

    thread = threading.Thread(target=_beep_thread)
    thread.daemon = True
    thread.start()

# Example usage with stop event:
def example_with_stop_paplay():
    sound_file = "/usr/share/sounds/freedesktop/stereo/bell.oga" #example path.
    stop_event = threading.Event()
    beep_thread = threading.Thread(target=beep_alarm_paplay, args=(sound_file, 10, 0.2, stop_event))
    beep_thread.daemon = True
    beep_thread.start()

    time.sleep(3)
    print("Stopping alarm...")
    stop_event.set()
    time.sleep(1)
    print("Alarm stopped (or should be).")

# Example usage without stop event (runs to completion):
def example_without_stop_paplay():
    sound_file = "/usr/share/sounds/freedesktop/stereo/bell.oga" #example path.
    beep_alarm_paplay(sound_file, 5, 0.7)
    print("Alarm finished.")

if __name__ == "__main__":
    print("Example with stop event:")
    example_with_stop_paplay()
    print("\nExample without stop event:")
    example_without_stop_paplay() 