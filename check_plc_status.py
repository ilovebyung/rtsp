import time
import threading

def check_status():
    """Simulates a function that might take time."""
    time.sleep(0.1)  # Simulate a process that sometimes exceeds the timeout
    return "Status checked"

def check_speed():
    """Monitors check_status() and skips it if it exceeds the timeout."""
    timeout = 0.1
    result = None
    event = threading.Event()

    def run_with_timeout():
        nonlocal result
        try:
            result = check_status()
        finally:
            event.set()

    thread = threading.Thread(target=run_with_timeout)
    thread.start()
    event.wait(timeout)

    if event.is_set():
        if result is not None:
            print("Status:", result)
        else:
            print("Status check completed within timeout, but no result returned") # in case check_status returns None.
    else:
        print("Status check timed out.")

# Example usage:
check_speed()
time.sleep(0.1) #give some time to the thread to finish, if it is going to finish.
check_speed()