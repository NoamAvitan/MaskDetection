import cv2
import RPi.GPIO as GPIO

from time import sleep
from threading import Thread
from objects import VideoStream, MaskObject, ScreenObject



class Detection(object):

    def __init__(self):
        self.mask_object = MaskObject()
        self.authorized_frames_count = 0
        self.unauthorized_frames_count = 0
        self.camera = VideoStream().start()

    def update_frame_type(self, pred_labels):
        if not pred_labels:
            print("[INFO] Person not detected")
            self.authorized_frames_count = 0
            self.unauthorized_frames_count = 0
        elif "No Mask" in pred_labels:
            print("[INFO] Person detect without Mask")
            self.authorized_frames_count = 0
            self.unauthorized_frames_count += 1
        else:
            print("[INFO] Person detect with Mask")
            self.unauthorized_frames_count = 0
            self.authorized_frames_count += 1

    @staticmethod
    def get_plug_camera_error():
        input('There is no camera device connected, Please plug a camera to the Raspberry Pi.\n ' \
              'Press any key to continue after you connect the cammera...')

    def unauthorized_frames_run(self):
        print("[INFO] 5 frame without Mask")
        self.camera.stop()
        self.unauthorized_frames_count = 0
        self.authorized_frames_count = 0
        ScreenObject.play_no_mask_video()
        self.camera = VideoStream().start()

    def authorized_frames_run(self):
        """
        Opening Door
        """
        print("[INFO] 5 frame with Mask")
        self.camera.stop()
        screen = Thread(target=ScreenObject.play_entry_allowed_screen, args=())
        screen.start()

        GPIO.setmode(GPIO.BOARD)

        # Set pin 11 as an output, and set servo1 as pin 11 as PWM
        GPIO.setup(11,GPIO.OUT)
        servo1 = GPIO.PWM(11,50) # Note 11 is pin, 50 = 50Hz pulse

        servo1.start(0)
        sleep(2)
        
        # open door
        duty = 9
        servo1.ChangeDutyCycle(duty)
        sleep(2)
        
        # close door
        duty = 12
        servo1.ChangeDutyCycle(duty)
        sleep(3)

        #Clean things up at the end
        servo1.stop()
        GPIO.cleanup()
        self.authorized_frames_count = 0
        self.unauthorized_frames_count = 0
        sleep(2)
        self.camera = VideoStream().start()

    def check_frame_counter_pass_5(self):
        if self.unauthorized_frames_count == 5:
            self.unauthorized_frames_run()
        elif self.authorized_frames_count == 5:
            self.authorized_frames_run()

    def detection(self):
        # Initialize webcam feed
        # camera = VideoStream().start()

        while True:
            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            if not self.camera.grabbed:
                self.get_plug_camera_error()
                continue  # skips the rest of the commands for the current loop

            frame = self.camera.read()

            # use the face detector model to extract bounding boxes and pre-processed faces from the frame
            locs, faces = self.mask_object.detect_faces(frame)

            # run mask prediction using our mask classification model and retrieve predicted label and confidence
            pred_labels, scores = self.mask_object.predict(faces)

            # loop over the detected face locations and their corresponding locations
            frame = ScreenObject.draw_boxes_with_predictions(frame, locs, pred_labels, scores)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Mask Detection', frame)
            cv2.moveWindow('Mask Detection', 0, 0)
            cv2.setWindowProperty('Mask Detection', cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

            self.update_frame_type(pred_labels=pred_labels)
            self.check_frame_counter_pass_5()

            # Press 'q' to break out of the loop and quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Clean up
        # video.release()
        self.camera.stop()
        cv2.destroyAllWindows()
