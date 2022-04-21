import cv2
import threading

from pydub import AudioSegment
from pydub.playback import play
from time import sleep, time

class ScreenObject(object):

    @staticmethod
    def play_no_mask_video():
        video_path = "/home/pi/MaskDetection/mask-instructions.mp4"
        audio_path = "/home/pi/MaskDetection/mask-instructions.aac"
        audio = AudioSegment.from_file(audio_path)

        def play_audio(audio):
            '''
            it takes opencv roughly a third of
            a second to start playing the video,
            so we delay the audio in order to
            synchronize it with the video.
            '''
            sleep(1 / 3)
            play(audio)

        t = threading.Thread(
            target=play_audio, args=(audio, )
        )  # we used threading in order to play the audio and video simultanously.
        t.start()

        frame_rate = 30
        time_before_frame = time()  # reset time
        cap = cv2.VideoCapture(video_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                time_elapsed = time() - time_before_frame
                if time_elapsed < 1. / frame_rate:  # making sure the playback frame rate isn't faster than it should be.
                    sleep((1. / frame_rate) - time_elapsed)  # if it is too fast, it will wait for the remaining time before loading the next frame.
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow('Mask Detection', frame)
                time_before_frame = time()
                cv2.waitKey(1)
            else:
                break
        cap.release()

    @staticmethod
    def play_entry_allowed_screen():
        entry_allowed_frame = cv2.imread("/home/pi/MaskDetection/images/OK.jpg")
        entry_allowed_frame = cv2.resize(entry_allowed_frame, (640, 480))
        cv2.imshow('Mask Detection', entry_allowed_frame)
        cv2.waitKey(5000)
    
    @staticmethod
    def draw_boxes_with_predictions(frame, locs, pred_labels, scores):
        for (box, pred_label, score) in zip(locs, pred_labels, scores):
            # get box position on screen
            (startY, startX, endY, endX) = box
            # choose color and starting label for box
            label = ''
            if 'No Mask' in pred_labels:
                label = 'No Mask'
                color = (0, 0, 255)
            else:
                label = 'Have Mask'
                color = (0, 255, 0)
            
            # add probability of prediction to label
            label += f' - {round(float(score)*100,2)}%'
            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        return frame
