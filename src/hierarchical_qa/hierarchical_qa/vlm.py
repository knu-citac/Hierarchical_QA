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
import json

from cv_bridge import CvBridge

class BLIP_ROS():
    def __init__(self):
        
        #specify the folder that contains the weights of the model
        self.weights_path = "./src/hierarchical_qa/hierarchical_qa/total92.pth"
        #self.device = "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        #self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        config = GenerationConfig(max_new_tokens=10, num_return_sequences=1)
        self.fmodel = torch.load(self.weights_path).to(self.device)
        setattr(self.fmodel.config.text_config, 'label_smoothing', False)
        self.model=  BlipForQuestionAnswering(self.fmodel.config).to(self.device)
        for name, para in self.model.named_parameters():
            for fname, fpara in self.fmodel.named_parameters():
                if name == fname:
                    para.data = fpara.data

        
        self.questions_ids = {
            0: "Is the ego vehicle moving on a straight road?",
            1: "Is the ego vehicle moving on a curved road?",
            2: "In which direction does the road curve?",
            
            3: "Is the ego vehicle moving through an intersection?",
            
            4: "Is the ego vehicle approaching an intersection?",
            5: "Is it possible for the ego vehicle to go straight at the intersection?",
            6: "Is it possible for the ego vehicle to turn left at the intersection?",
            7: "Is it possible for the ego vehicle to turn right at the intersection?",
            
            8: "Is there a vehicle in front of the ego vehicle?",
            9: "Is the ego vehicle close to the vehicle in front?",
            
            10: "Is there a vehicle moving through the intersection?",
            11: "Is the vehicle moving through the intersection potentially crossing path with the ego vehicle?",
            
            12: "Is there a vehicle approaching the intersection from the left?",
            13: "Is the vehicle approaching the intersection from the left potentially crossing path with the ego vehicle if the ego vehicle goes straight?",
            14: "Is the vehicle approaching the intersection from the left potentially crossing path with the ego vehicle if the ego vehicle turns left?",
            15: "Is the vehicle approaching the intersection from the left potentially crossing path with the ego vehicle if the ego vehicle turns right?",
            
            16: "Is there a vehicle approaching the intersection from the right?",
            17: "Is the vehicle approaching the intersection from the right potentially crossing path with the ego vehicle if the ego vehicle goes straight?",
            18: "Is the vehicle approaching the intersection from the right potentially crossing path with the ego vehicle if the ego vehicle turns left?",
            19: "Is the vehicle approaching the intersection from the right potentially crossing path with the ego vehicle if the ego vehicle turns right?",
            
            
            20: "Is there a vehicle approaching the intersection from the opposite direction?",
            21: "Is the vehicle approaching the intersection from the opposite direction potentially crossing path with the ego vehicle if the ego vehicle goes straight?",
            22: "Is the vehicle approaching the intersection from the opposite direction potentially crossing path with the ego vehicle if the ego vehicle turns left?",
            23: "Is the vehicle approaching the intersection from the opposite direction potentially crossing path with the ego vehicle if the ego vehicle turns right?",
            
            24: "Is there a pedestrian in front of the ego vehicle?",
            25: "Is the ego vehicle close to the pedestrian in front?",
            26: "Is the pedestrian in front of the ego vehicle potentially crossing path with the ego vehicle if the ego vehicle continues moving in the same path?",
            
            27: "Is there a pedestrian near the road in front of the ego vehicle?",
            28: "Is the pedestrian near the road in front of the ego vehicle potentially crossing path with the ego vehicle if the ego vehicle continues moving in the same path?",
            
            29: "Is there a pedestrian on the road to the left of the intersection?",
            30: "Is the pedestrian on the road to the left of the intersection potentially crossing path with the ego vehicle if the ego vehicle turns left?",
            31: "Is there a pedestrian on the road to the right of the intersection?",
            32: "Is the pedestrian on the road to the right of the intersection potentially crossing path with the ego vehicle if the ego vehicle turns right?",
            
            33: "Are there any traffic signs in the scene?",
            34: "What is the type of the traffic sign in the scene?",
            
            35: "Are there any crosswalks in the scene?",
            36: "Is the ego vehicle approaching the crosswalk",
            37: "Is the ego vehicle close to the crosswalk?",
            
            38: "Are there any speed bumps in the scene?",
            39: "Is the ego vehicle approaching the speed bump?",
            40: "Is the ego vehicle close to the speed bump?",
        }


        self.explanations = {
            0: "the ego vehicle is moving on a straight road",
            1: "the ego vehicle is traveling on a curved road",
            2: ["the ego vehicle is moving on a road that curves to the left", "The ego vehicle is moving on a road that curves to the right"],
            
            3: "the ego vehicle is moving through an intersection",
            
            4: "the ego vehicle is approaching an intersection",
            5: " straight",
            6: " left",
            7: " right",
            
            8: "there is a vehicle in front of the ego vehicle",
            9: "the ego vehicle is close to the vehicle in front",
            
            10: "there is a vehicle moving through the intersection",
            11: "the vehicle moving through the intersection is potentially crossing path with the ego vehicle",
            
            12: "there is a vehicle approaching the intersection from the left", # is potentially crossing path with the ego vehicle",# if the ego vehicle goes",
            13: " straight",
            14: " left",
            15: " right",
            
            16: "there is a vehicle approaching the intersection from the right",# is potentially crossing path with the ego vehicle",# if the ego vehicle goes",
            17: " straight",
            18: " left",
            19: " right",
            
            
            20: "there is a vehicle approaching the intersection from the opposite direction", # is potentially crossing path with the ego vehicle",# if the ego vehicle goes",
            21: " straight",
            22: " left",
            23: " right",
            
            24: "there is a pedestrian in front of the ego vehicle",
            25: "the ego vehicle is close to the pedestrian in front",
            26: "the pedestrian in front is potentially crossing path with the ego vehicle if the ego vehicle continues moving in the same path",
            
            27: "there is a pedestrian near the road in front of the ego vehicle",
            28: "the pedestrian near the road is potentially crossing path with the ego vehicle if the ego vehicle continues moving in the same path",
            
            29: "there is a pedestrian to the left of the road",
            30: "the pedestrian on the road to the left of the intersection is potentially crossing path with the ego vehicle if the ego vehicle turns left",
            31: "there is a pedestrian to the right of the road",
            32: "the pedestrian on the road to the right of the intersection is potentially crossing path with the ego vehicle if the ego vehicle turns right",
            
            33: None,
            34: ["there is a bicycle crossing traffic sign in the scene", "there is a pedestrian crossing traffic sign in the scene"],
            
            35: None,
            36: "the ego vehicle is approaching a crosswalk",
            37: "the ego vehicle is close to the crosswalk",
            
            38: None,
            39: "the ego vehicle is approaching a speed bump",
            40: "the ego vehicle is close to the speed bump",

            192: "the vehicles approaching the intersection from the left and right are potentially crossing path with the ego vehicle",
            240: "the vehicles approaching the intersection from the left and the opposite direction are potentially crossing path with the ego vehicle",
            320: "the vehicles approaching the intersection from the right and the opposite direction are potentially crossing path with the ego vehicle",
            3840: "the vehicles approaching the intersection from the left, right, and opposite directions are potentially crossing path with the ego vehicle",
            
            899: "there are pedestrians to the left and right of the road",

            960: "the pedestrians on the roads to the left and right of the intersection are potentially crossing path with the ego vehicle",
            
            1404: "the ego vehicle is approaching a speed bump and a crosswalk",#
            1480: "the ego vehicle is approaching a speed bump and a crosswalk",#"the ego vehicle is close to a speed bump and a crosswalk",
            1440: "the ego vehicle is approaching a speed bump and a crosswalk",#"the ego vehicle is close to a speed bump and approaching a crosswalk",
            1443: "the ego vehicle is approaching a speed bump and a crosswalk",#"the ego vehicle is close to a crosswalk and approaching a speed bump",
        }

        self.possible_answers = {0: ["Yes", "No", "None"],
                            1: ["Yes", "No", "None"],
                            2: ["Left", "Right", "None"],
                            3: ["Yes", "No", "None"],
                            4: ["Yes", "No", "None"],
                            5: ["Yes", "No", "None"],
                            6: ["Yes", "No", "None"],
                            7: ["Yes", "No", "None"],
                            8: ["Yes", "No", "None"],
                            9: ["Yes", "No", "None"],
                            10: ["Yes", "No", "None"],
                            11: ["Yes", "No", "None"],
                            12: ["Yes", "No", "None"],
                            13: ["Yes", "No", "None"],
                            14: ["Yes", "No", "None"],
                            15: ["Yes", "No", "None"],
                            16: ["Yes", "No", "None"],
                            17: ["Yes", "No", "None"],
                            18: ["Yes", "No", "None"],
                            19: ["Yes", "No", "None"],
                            20: ["Yes", "No", "None"],
                            21: ["Yes", "No", "None"],
                            22: ["Yes", "No", "None"],
                            23: ["Yes", "No", "None"],
                            24: ["Yes", "No", "None"],
                            25: ["Yes", "No", "None"],
                            26: ["Yes", "No", "None"],
                            27: ["Yes", "No", "None"],
                            28: ["Yes", "No", "None"],
                            29: ["Yes", "No", "None"],
                            30: ["Yes", "No", "None"],
                            31: ["Yes", "No", "None"],
                            32: ["Yes", "No", "None"],
                            33: ["Yes", "No", "None"],
                            34: ["Bicycle", "Pedestrian crossing", "None"],
                            35: ["Yes", "No", "None"],
                            36: ["Yes", "No", "None"],
                            37: ["Yes", "No", "None"],
                            38: ["Yes", "No", "None"],
                            39: ["Yes", "No", "None"],
                            40: ["Yes", "No", "None"],
                            }
        


    def road_features (self, answers, image):
        temp_ans = self.infer_answer(image, 33)
        answers[33] = self.possible_answers[33][temp_ans-1]
        if answers[33] == 'Yes':
            temp_ans = self.infer_answer(image, 34)
            answers[34] = self.possible_answers[34][temp_ans-1]
        else:
            answers[34] = self.possible_answers[34][3-1]

        ################## Crosswalks
        temp_ans = self.infer_answer(image, 35)
        answers[35] = self.possible_answers[35][temp_ans-1]
        if answers[35] == 'Yes':
            temp_ans = self.infer_answer(image, 37)
            answers[37] = self.possible_answers[37][temp_ans-1]
            answers[36] = self.possible_answers[36][temp_ans+1-1 if temp_ans==1 else temp_ans-1-1]
            
        else:
            answers[37] = self.possible_answers[37][3-1]
            answers[36] = self.possible_answers[36][3-1]
        
        
        ################## Speedbump
        temp_ans = self.infer_answer(image, 38)
        answers[38] = self.possible_answers[38][temp_ans-1]
        if answers[38] == 'Yes':

            temp_ans = self.infer_answer(image, 40)
            answers[40] = self.possible_answers[40][temp_ans-1]
            answers[39] = self.possible_answers[39][temp_ans+1-1 if temp_ans==1 else temp_ans-1-1]
            
        else:
            answers[39] = self.possible_answers[39][3-1]
            answers[40] = self.possible_answers[40][3-1]
        
        return answers


    def is_there_a_pedestrian_on_right_or_left_intersection  (self, answers, q_id, image, applicable= True):
        if not applicable:
            answers[q_id] = self.possible_answers[q_id][3-1]
            answers[q_id+1] = self.possible_answers[q_id+1][3-1]
            return answers
        
        temp_ans = self.infer_answer(image, q_id)
        answers[q_id] = self.possible_answers[q_id][temp_ans-1]

        if answers[q_id] == 'Yes':
            temp_ans = self.infer_answer(image, q_id)
            answers[q_id+1] = self.possible_answers[q_id+1][temp_ans-1]
        else:
            answers[q_id+1] = self.possible_answers[q_id+1][3-1]
        

        return answers

    def pedestrains (self, answers, image):
        
        #Q 24 25 26 27 28

        temp_ans = self.infer_answer(image, 24)
        answers[24] = self.possible_answers[24][temp_ans-1]

        if answers[24] == 'Yes':
            
            answers[27] = self.possible_answers[27][3-1]
            answers[28] = self.possible_answers[28][3-1]

            temp_ans = self.infer_answer(image, 25)
            answers[25] = self.possible_answers[25][temp_ans-1]
            
            if answers[25] == 'No':
                temp_ans = self.infer_answer(image, 26)
                answers[26] = self.possible_answers[26][temp_ans-1]
                
            else:
                answers[26] = self.possible_answers[26][3-1]
            
        else:
            
            answers[25] = self.possible_answers[25][3-1]
            answers[26] = self.possible_answers[26][3-1]

            temp_ans = self.infer_answer(image, 27)
            answers[27] = self.possible_answers[27][temp_ans-1]

            if answers[27] == 'Yes':
            
                temp_ans = self.infer_answer(image, 28)
                answers[28] = self.possible_answers[28][temp_ans-1]
            
            else:
                
                answers[28] = self.possible_answers[28][3-1]
                
        return answers



    def is_there_a_vehicle  (self, answers, image):
        
        temp_ans = self.infer_answer(image, 8)
        answers[8] = self.possible_answers[8][temp_ans-1]

        if answers[8] == 'Yes':
            temp_ans = self.infer_answer(image, 9)
            answers[9] = self.possible_answers[9][temp_ans-1]
            
        else:
            answers[9] = self.possible_answers[9][3-1]
            
        return answers
    
    def is_a_vehicle_approaching_an_intersection (self, answers, q_id, image, applicable= True):
        
        if not applicable:
            answers[q_id] = self.possible_answers[q_id][3-1]
            answers[q_id+1] = self.possible_answers[q_id+1][3-1]
            answers[q_id+2] = self.possible_answers[q_id+2][3-1]
            answers[q_id+3] = self.possible_answers[q_id+3][3-1]
            
            return answers
        
        temp_ans = self.infer_answer(image, q_id)
        answers[q_id] = self.possible_answers[q_id][temp_ans-1]
        if answers[q_id] == 'Yes': #there is a car approaching from this direction
            if answers[5] == 'Yes': #in case ego vehicle wants to go straight, check if possible
                temp_ans = self.infer_answer(image, q_id)
                answers[q_id+1] = self.possible_answers[q_id+1][temp_ans-1]
            else:
                answers[q_id+1] = self.possible_answers[q_id+1][3-1]
            
            if answers[6] == 'Yes': #in case ego vehicle wants to go left, check if possible
                temp_ans = self.infer_answer(image, q_id)
                answers[q_id+2] = self.possible_answers[q_id+2][temp_ans-1]
            else:
                answers[q_id+2] = self.possible_answers[q_id+2][3-1]
            
            if answers[7] == 'Yes': #in case ego vehicle wants to go right, check if possible
                temp_ans = self.infer_answer(image, q_id)
                answers[q_id+3] = self.possible_answers[q_id+3][temp_ans-1]
            else:
                answers[q_id+3] = self.possible_answers[q_id+3][3-1]
            
        else:
            answers[q_id+1] = self.possible_answers[q_id+1][3-1]
            answers[q_id+2] = self.possible_answers[q_id+2][3-1]
            answers[q_id+3] = self.possible_answers[q_id+3][3-1]
        
        return answers
    

    def approaching_an_intersection (self, answers, image, applicable= True):
        if not applicable:
            answers[4] = self.possible_answers[4][3-1]
            answers[5] = self.possible_answers[5][3-1]
            answers[6] = self.possible_answers[6][3-1]
            answers[7] = self.possible_answers[7][3-1]
            answers[10] = self.possible_answers[10][3-1]
            answers[11] = self.possible_answers[11][3-1]
            answers[12] = self.possible_answers[12][3-1]
            answers[13] = self.possible_answers[13][3-1]
            answers[14] = self.possible_answers[14][3-1]
            answers[15] = self.possible_answers[15][3-1]
            answers[16] = self.possible_answers[16][3-1]
            answers[17] = self.possible_answers[17][3-1]
            answers[18] = self.possible_answers[18][3-1]
            answers[19] = self.possible_answers[19][3-1]
            answers[20] = self.possible_answers[20][3-1]
            answers[21] = self.possible_answers[21][3-1]
            answers[22] = self.possible_answers[22][3-1]
            answers[23] = self.possible_answers[23][3-1]
            answers = self.is_there_a_pedestrian_on_right_or_left_intersection(answers, 29, image, applicable= False)
            answers = self.is_there_a_pedestrian_on_right_or_left_intersection(answers, 31, image, applicable= False)
            
            return answers
        
        
        temp_ans = self.infer_answer(image, 4)
        answers[4] = self.possible_answers[4][temp_ans-1]
        
        if answers[4] == 'Yes': #Aproaching Intersection
        
        
            temp_ans = self.infer_answer(image, 10) 
            answers[10] = self.possible_answers[10][temp_ans-1]
            if answers[10] == 'Yes':
                temp_ans = self.infer_answer(image, 11) 
                answers[11] = self.possible_answers[11][temp_ans-1]
            else:
                answers[11] = self.possible_answers[11][3-1]

            temp_ans = self.infer_answer(image, 5) #straight
            answers[5] = self.possible_answers[5][temp_ans-1]

            temp_ans = self.infer_answer(image, 6) #left
            answers[6] = self.possible_answers[6][temp_ans-1]


            temp_ans = self.infer_answer(image, 7) #right
            answers[7] = self.possible_answers[7][temp_ans-1]
            
            if answers[6] == 'Yes':
                answers = self.is_a_vehicle_approaching_an_intersection(answers, 12, image)
            else:
                answers = self.is_a_vehicle_approaching_an_intersection(answers, 12, image, applicable= False)
            if answers[7] == 'Yes':    
                answers = self.is_a_vehicle_approaching_an_intersection(answers, 16, image)
            else:
                answers = self.is_a_vehicle_approaching_an_intersection(answers, 16, image, applicable= False)    
            if answers[5] == 'Yes':
                answers = self.is_a_vehicle_approaching_an_intersection(answers, 20, image)
            else:
                answers = self.is_a_vehicle_approaching_an_intersection(answers, 20, image, applicable= False)
                
            answers = self.is_there_a_pedestrian_on_right_or_left_intersection(answers, 29, image)
            answers = self.is_there_a_pedestrian_on_right_or_left_intersection(answers, 31, image)
            
        else:
            answers = self.approaching_an_intersection (answers, image, applicable= False)
        
        return answers

    def stright_curved_intersection(self, answers, image):

        temp_ans = self.infer_answer(image, 0)
        answers[0] = self.possible_answers[0][temp_ans-1]
        
        
        if  answers[0] == 'Yes':  #Straight
            answers[1] = self.possible_answers[1][2-1]
            answers[2] = self.possible_answers[2][3-1]
            answers[3] = self.possible_answers[3][2-1]

        
        else:  #Not Straight
            temp_ans = self.infer_answer(image, 1)
            answers[1] = self.possible_answers[1][temp_ans-1]
            
            if answers[1] == 'Yes':  #Curved
                temp_ans = self.infer_answer(image, 2)
                
                answers[2] = self.possible_answers[2][temp_ans-1]
                answers[3] = self.possible_answers[3][2-1]

            else: #Not Curved #Inside Intersection

                answers[2] = self.possible_answers[2][3-1]
                
                
                temp_ans = self.infer_answer(image, 3)
                answers[3] = self.possible_answers[3][temp_ans-1]

                #answers[3] = self.possible_answers[3][1-1]
                
        return answers    

    def infer_answer(self, image, q_id):
        
        inputs = self.processor(images=image, text=self.questions_ids[q_id], return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        outputs = self.model.generate(**inputs)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)

        if answer.lower() == self.possible_answers[q_id][0].lower():
            return 1
        elif answer.lower() == self.possible_answers[q_id][1].lower():
            return 2
        else:
            return 3
        


    def convert_answers_to_id(self, answers):
        a_id = {"yes": 0, "no": 1, "none": 2, "left":0, "right":1, "bicycle": 0, "pedestrian crossing":1}
        for k, v in answers.items():
            answers[k] = a_id[v.lower()]
        return answers
    
    
    def provide_explanations(self, answers):
         
        def capitalize_and_fullstop (sen):
            if not sen == "":
                ntemp = ""
                ntemp += sen[0].capitalize()
                ntemp += sen[1:]
                ntemp+=". "
            else:
                ntemp = ""

            return ntemp

        answers = self.convert_answers_to_id(answers)
    
        #assert answers[0] == 0 or answers[3] == 0 or answers[2] < 2 or answers[1] == 0, "The ego vehicle must be moving either on a straight road, curved road, or through an intersection" + str(answers[0]) + str(answers[3]) +str(answers[1])+ str(answers[2] )
       
        sentence = ""
        for k, v in self.explanations.items():
            
            if v == None or k>41:
                continue

            elif k in [2, 5,6,7,9,11, 13,14,15,16,17,18,19,20, 21,22,23,   28,30, 31, 32, 37,39,40]:
                continue

            elif isinstance(v, list):
                temp = v[answers[k]] if answers[k] != 2 else ""
                
        
            #elif k in [4]:
            #    temp = v if answers[k] == 0 else ""
            #    temp = capitalize_and_fullstop(temp)

            elif k in [1]:
                if answers[2] < 2:
                    temp = self.explanations[2][answers[2]]
                elif answers[1] == 0:
                    temp = self.explanations[1]
                else:
                    temp = ""
            
            elif k in [12]:
                if answers[12] == 0 and  answers[16] == 0 and answers[20] == 0:
                    temp = self.explanations[12*16*20]
                elif answers[12] == 0 and  answers[16] == 0:
                    temp = self.explanations[12*16]
                elif answers[12] == 0 and answers[20] == 0:
                    temp = self.explanations[12*20]
                elif answers[16] == 0 and answers[20] == 0:
                    temp = self.explanations[16*20]
                elif answers[12] == 0:
                    temp = self.explanations[12]
                elif answers[16] == 0:
                    temp = self.explanations[16]
                elif answers[20] == 0:
                    temp = self.explanations[20]
                else:
                    temp = ""
                    

            elif k in [10]:
                if answers[11] == 0:
                    temp = self.explanations[11]
                elif answers[10] == 0:
                    temp = self.explanations[10]
                else:
                    temp = ""   
                
            
            #elif k in [30]:
            #    if answers[30] == 0 and  answers[32] == 0:
            #        temp = self.explanations[30*32]
            #    elif answers[30] == 0:
            #        temp = self.explanations[30]
            #    elif answers[32] == 0:
            #        temp = self.explanations[32] 
            #    else:
            #        temp = ""
                    

                
            elif k in [29]:
                if answers[29] == 0 and  answers[31] == 0:
                    temp = self.explanations[29*31]
                elif answers[29] == 0:
                    temp = self.explanations[29]
                elif answers[31] == 0:
                    temp = self.explanations[31] 
                else:
                    temp = ""

            elif k in [36]:
                if answers[36] == 0 and  answers[39] == 0:
                    temp = self.explanations[36*39]
                elif answers[37] == 0 and  answers[40] == 0:
                    temp = self.explanations[37*40]

                elif answers[36] == 0 and  answers[40] == 0:
                    temp = self.explanations[36*40]

                elif answers[37] == 0 and  answers[39] == 0:
                    temp = self.explanations[37*39] 
                elif answers[36] == 0:
                    temp = self.explanations[36]
                elif answers[39] == 0:
                    temp = self.explanations[39]

                elif answers[37] == 0:
                    temp = self.explanations[36] 
                elif answers[40] == 0:
                    temp = self.explanations[39]
                else:
                    temp = ""
                    #temp =  self.explanations[36*39]

                
                


            elif k in [8]:
                if answers[9] == 0:
                    temp = self.explanations[9]
                elif answers[8] == 0:
                    temp = self.explanations[8]
                else:
                    temp = ""
                    
                
                

            elif k in [27]:

                if answers[28] == 0:# and not (answers[30] == 0 or  answers[32] == 0):
                    temp = self.explanations[28]
                elif answers[27] == 0:# and not (answers[30] == 0 or  answers[32] == 0):
                    temp = self.explanations[27]
                else:
                    temp = ""

            
                
                  

            else:
                temp = v if answers[k] == 0 else ""
                
            
            temp = capitalize_and_fullstop(temp)
            sentence += temp

        
        return sentence    

    def inference(self, image):
        
        
        answers = {}
        
        answers = self.stright_curved_intersection(answers, image)
        
        if answers[3] == "Yes": ## moving through intersection, so it can not approach an intersection
            answers = self.approaching_an_intersection (answers, image, applicable= False)
        else:
            answers = self.approaching_an_intersection (answers, image, applicable= True)
        answers = self.pedestrains(answers, image)
        answers = self.road_features(answers, image)
        answers = self.is_there_a_vehicle(answers, image)
        
        
        sen = self.provide_explanations(answers)
        
        return sen, answers


    def pass_frame_to_blip (self, frame):
        
        sen, answers = self.inference(frame)
        return sen, answers
    
            
    # if frame_counter == stop_frame-1:
    #     frameImageFileName = str(f'/home/safaa/Desktop/Random Files/240111_dashcam/image{frame_counter}.png')
    #     cv2.imwrite(frameImageFileName, frame)




class VLM(Node):

    def __init__(self):
        super().__init__('VLM')

        self.vlm_in_background = True
        self.save_images = False
        self.frames_topic = "/flir_camera/image_raw"
        timer_period = 0.1
        self.counter = 0
        self.vlm_counter = 0

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.subscription = self.create_subscription(Image, self.frames_topic, 
                                                self.listener_callback, qos_profile)
        
        self.vlm_publisher = self.create_publisher(String, "/vlm", 10)
        self.vlm_publisher_timer = self.create_timer(timer_period, self.vlm_timer_callback)

        self.answers_publisher = self.create_publisher(String, "/vlm_answers", 10)
        self.answers_publisher_timer = self.create_timer(timer_period, self.answers_timer_callback)

        

        self.last_answers = None
        self.last_expln = ""

        self.vlm_text_duration = 2



        self.vlm_text_timer = time.time()

        self.blip_ros = BLIP_ROS()
        self.bridge = CvBridge()

        
    def listener_callback(self, msg):
        #self.get_logger().info("------------------------- Image Received")
        height = msg.height
        width = msg.width
        channel = msg.step//msg.width


        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        expln, answers = self.blip_ros.pass_frame_to_blip(frame)
        self.get_logger().info(expln)
        
        self.last_answers = answers

        timer_bool, self.vlm_text_timer, time_diff  = self.timer(self.vlm_text_timer , self.vlm_text_duration)
        if timer_bool:
            
            self.last_expln = expln
            
        
        
        self.get_logger().info("----------- Scene understanding process is complete")
        
        

    def timer (self, timer, duration):
        now = time.time()
        time_diff = now - timer
        if time_diff>duration:
            timer = time.time()
            return True, timer, time_diff
        return False, timer, time_diff
    

    def vlm_timer_callback(self):

        
        self.vlm_publisher.publish(String(data=self.last_expln))
        #self.get_logger().info('VLM Output Published %d' % self.vlm_counter)
        
        # image counter increment
        self.vlm_counter += 1
        
        return None

    def answers_timer_callback(self):

        if not self.last_answers == None:
            self.answers_publisher.publish(String(data=str(self.last_answers)))
            #self.get_logger().info('VLM Answers Published ' + str(self.last_answers))



def main(args=None):
    rclpy.init(args=args)
    vlm = VLM()
    rclpy.spin(vlm)

    vlm.destroy_node()
    rclpy.shutdown()

    cv2.destroyAllWindows()
    return None

if __name__ == '__main__':
    main()