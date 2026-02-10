import numpy as np
import matplotlib.pyplot as plt
import pygame as pyg
import quiz_Utils as qu
import sys
import os
import importlib
importlib.reload(qu) 

os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"

pyg.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700

screen = pyg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pyg.display.set_caption("Graph quiz")

font = pyg.font.SysFont(None, 36)
small_font = pyg.font.SysFont(None, 28)

White = (255, 255, 255)
Black = (0, 0, 0)
Green = (0, 200, 0)
Red = (200, 0, 0)


options = ["Integers", "Uniform", "Normal", "Binomial", "Negative binomial", "Gamma", "Geometric"]

key_map = {
    pyg.K_1: options[0],
    pyg.K_2: options[1],
    pyg.K_3: options[2],
    pyg.K_4: options[3],
    pyg.K_5: options[4],
    pyg.K_6: options[5],
    pyg.K_7: options[6],
}

run = True
correct_answer, hist_image, hist_rect = qu.new_question(SCREEN_WIDTH)
feedback = ""
feedback_color = Black
waiting_for_next = False

while run == True:

    screen.fill(Black)
    screen.blit(hist_image, hist_rect.move(100, 130))

    
    title = font.render("Which distribution is this?", True, White)
    screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))

    for i, option in enumerate(options):
        text = small_font.render(f"{i+1}. {option}", True, White)
        screen.blit(text, (50, 350 + i * 30))

    if feedback:
        msg = font.render(feedback, True, feedback_color)
        screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2, 540))

    pyg.display.flip()



    for event in pyg.event.get():
        if event.type == pyg.QUIT:
            run = False
        
        if event.type == pyg.KEYDOWN:
            if not waiting_for_next and event.key in key_map:
                if key_map[event.key] == correct_answer:
                    feedback = "Correct!"
                    feedback_color = Green
                else:
                    feedback =f"Wrong! Correct answer was {correct_answer}"
                    feedback_color = Red
                waiting_for_next = True
                
            elif waiting_for_next and event.key == pyg.K_SPACE:
                correct_answer, hist_image, hist_rect = qu.new_question(SCREEN_WIDTH)
                feedback = ""
                waiting_for_next = False

                
            


pyg.quit()
sys.exit()