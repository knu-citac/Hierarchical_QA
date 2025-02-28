#!/usr/bin/env python3

#___Import Modules:
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


from transformers import AutoProcessor, BlipForQuestionAnswering
import torch
from transformers import pipeline, set_seed, GenerationConfig

from rclpy.qos import QoSProfile, QoSHistoryPolicy , QoSReliabilityPolicy

from PIL import ImageFont, ImageDraw
from PIL import Image as PILImage

from textwrap import wrap
from gtts import gTTS
import os

import time
from cv_bridge import CvBridge


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        
        self.save_images = True
        self.frames_topic = "/flir_camera/image_raw"
        self.counter = 0
        self.debugging = False

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.subscription_emo = self.create_subscription(String, '/emotion_detected', 
                                                self.listener_callback_emotion, 10)
        
        
        self.subscription_frame = self.create_subscription(Image, self.frames_topic, 
                                                self.listener_callback, qos_profile)
        

        self.subscription_vlm = self.create_subscription(String, "/vlm", 
                                                self.listener_callback_vlm, 10)

        self.subscription_vlm_answers = self.create_subscription(String, "/vlm_answers", 
                                                self.listener_callback_vlm_answers, 10)
        
        
        
        self.negative_emotion_detected =[False, ""]
        self.font = ImageFont.truetype("./src/hierarchical_qa/hierarchical_qa/font/ABeeZee-Regular.otf", 25)
        self.font.size = 200
        self.last_answers = None
        self.last_expln = ""

        self.emotion_text_duration = 6
        self.emotion_audio_duration = 8

        self.emotion_text_timer = time.time()
        self.emotion_audio_timer = time.time() - self.emotion_audio_duration - 2

        self.audio = False
        self.bridge = CvBridge()


    def display_image(self, frame):
        expln = self.last_expln
        new_height = 750
        new_width = int(new_height * 1.6)
        original_frame = frame
        frame = PILImage.fromarray(frame)
        frame = frame.resize((new_width, new_height))
        draw = ImageDraw.Draw(frame)
        expln='\n'.join(wrap(expln, width=97))
        no_lines = expln.count("\n") 
        Y_start_point = 580
        dynamic_height = Y_start_point + (no_lines +1) * 25 +13 
        draw.text((20, Y_start_point),expln,(255,255,255),font=self.font)

        if self.negative_emotion_detected[0]:
            emo_expln = '\n'.join(wrap(self.negative_emotion_detected[1], width=100))
            draw.text((200, int(new_height * 0.5)),emo_expln,(129,231,255),font=self.font)
        
        if self.audio:
            self.speak_gtts(emo_expln)
            self.emotion_audio_timer = time.time()
            self.audio = False
            
        timer_bool, self.emotion_text_timer, time_diff = self.timer(self.emotion_text_timer, self.emotion_text_duration)
        if timer_bool:
            self.negative_emotion_detected[0] = False    
        #else:
            #self.get_logger().info("------ Emotion Message Timer ------" + str(time_diff))

        opencvImage = np.array(frame)#, cv2.COLOR_RGB2BGR)
        original_frame = np.array(original_frame)
        shapes = np.zeros_like(opencvImage, np.uint8)
        # Draw shapes
        if not expln == "":
            cv2.rectangle(shapes, (0, Y_start_point), (new_width, dynamic_height), (113, 125, 126), cv2.FILLED)

        ############# Compose
        alpha = 0.5
        mask = shapes.astype(bool)
        opencvImage[mask] = cv2.addWeighted(opencvImage, alpha, shapes, 1 - alpha, 0)[mask]


        cv2.imshow('image',opencvImage)
        cv2.waitKey(1)
        if self.save_images:
            file_name = "results/"+str(self.counter)+".png"
            cv2.imwrite(file_name, opencvImage)
            file_name = "results/original frames/"+str(self.counter)+".png"
            cv2.imwrite(file_name, original_frame)



    def listener_callback_vlm(self, msg):
        self.get_logger().info("----------- Received VLM message is "+str(msg.data))
        #self.get_logger().info("----------- VLM is working in background")
        self.last_expln = msg.data
        
    def listener_callback_vlm_answers(self, msg):
        #self.get_logger().info("----------- VLM is working in background")
        self.last_answers = eval(msg.data)
     


        
    def listener_callback(self, msg):
        #self.get_logger().info("----------- Frame is Received")
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.display_image(frame)
        self.counter += 1


    def speak_gtts(self, text):
        # Generate TTS output
        tts = gTTS(text=text, lang="en")
        # Save the audio to a file
        file_name = "temp.mp3"
        tts.save(file_name)
        # Play the audio file
        os.system(f"xdg-open {file_name}")


    def timer (self, timer, duration):
        now = time.time()
        time_diff = now - timer
        if time_diff>duration:
            timer = time.time()
            return True, timer, time_diff
        return False, timer, time_diff

    def listener_callback_emotion(self, msg):
        self.get_logger().info("----------- Received emotion message is "+str(msg.data))
        list_of_emotions = ['angry', 'disgust',  'surprise', 'fear', 'sad', 'happy', 'neutral']

        
        if msg.data in list_of_emotions[0:4]:
            msg_emotion = True
        else: 
            msg_emotion = False
        if msg_emotion and not self.last_answers == None:


            new_emotion_detected = False
            if self.debugging:
                self.get_logger().info("*******************************")


            if 0 in [self.last_answers[9]]:
                self.negative_emotion_detected[1] = "My apologies for the sudden stop; a vehicle is obstructing the road."
                self.negative_emotion_detected[0] = True
                new_emotion_detected = True
            elif 0 in [self.last_answers[24], self.last_answers[27]]:
                self.negative_emotion_detected[1] = "My apologies for the sudden stop; a pedestrian is crossing the road."
                self.negative_emotion_detected[0] = True
                new_emotion_detected = True
            #else:
            #    #return None
            #    expln = "My apologies for the sudden stop; something seems to be wrong."
            #    detected = False
            if self.negative_emotion_detected[0] and new_emotion_detected:
                self.get_logger().info(str("******** "+self.negative_emotion_detected[1]))
                self.emotion_text_timer = time.time()

                timer_bool, self.emotion_audio_timer, time_diff = self.timer(self.emotion_audio_timer, self.emotion_audio_duration)
                self.audio = True if timer_bool else False
                
                #self.get_logger().info("----------- Emotion message is processed")




def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)

    image_subscriber.destroy_node()
    rclpy.shutdown()

    cv2.destroyAllWindows()
    return None

if __name__ == '__main__':
    main()