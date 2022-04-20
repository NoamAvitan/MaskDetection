import cv2
import threading


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    """https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588"""
    def __init__(self, resolution=(640, 480)):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0, cv2.CAP_V4L)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC,
                              #cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3, resolution[0])
        #ret = self.stream.set(4, resolution[1])
        self.stream.set(3,640)#width
        self.stream.set(4,480)#height
        self.stream.set(10,100)#brightness

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True