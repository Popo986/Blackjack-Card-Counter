# ----- BLACKJACK CARD COUNTER ----- #
# Matthew Pope and Kate Vasquez (much based on Evan Juras)
# 9/16/23

import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


# --- INITIALIZATION --- #

# Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10

# Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera
videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),1).start()

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')

# Card counter variables
seenCards = []
card_count = 0
low_cc = ['Two', 'Three', 'Four', 'Five', 'Six']
high_cc = ['Ten', 'Jack', 'Queen', 'King', 'Ace']


# --- MAIN LOOP --- #

while True:
    time.sleep(.1)  # slows framerate

    image = videostream.read()  # Grab frame from video stream

    pre_proc = Cards.preprocess_image(image)  # Pre-process camera image (gray, blur, and threshold it)

    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)  # Find and sort the contours of all cards in the image

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:

                # Create a card object from the contour and append it to the list of cards
                # generates a flattened 200x300 image of the card, and isolates the card's suit and rank from the image.
                cards.append(Cards.preprocess_card(cnts_sort[i],image))

                # Find the best rank and suit match for the card.
                cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)

                uniqueCard = (cards[k].best_rank_match,cards[k].best_suit_match)  # creates temp card

                # Adds temp card to list of seen cards if new
                if(uniqueCard not in seenCards) and (uniqueCard[1] != 'Unknown') and (uniqueCard[0] != 'Unknown'):
                    seenCards.append(uniqueCard)

                card_count = 0  # resets variable before recount

                for j in range(len(seenCards)):  # calculates total running count
                    if seenCards[j][0] in low_cc:
                        card_count += 1
                    elif seenCards[j][0] in high_cc:
                        card_count -= 1

                # Draw center point and match result on the image.
                image = Cards.draw_results(image, cards[k])
                k = k + 1

        # Draw card contours on image
        if len(cards) != 0:
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)

    # Draw card count in the corner of the image
    cv2.putText(image,"Running Count: "+str(card_count),(10,26),font,1.2,(255,0,255),2,cv2.LINE_AA)

    # Finally, display the image with the identified cards!
    cv2.imshow("Card Detector",image)

    if cv2.waitKey(1) == ord(' '):
        break

# Close all windows and close the PiCamera video stream.
cv2.destroyAllWindows()
videostream.stop()

# !!! REMOVE BEFORE SUBMISSION !!! #
print(seenCards)
print(card_count)
