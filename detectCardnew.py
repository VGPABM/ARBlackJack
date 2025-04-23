import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import os
import ModulKlasifikasiCitraCNN as mCNN

cap = cv2.VideoCapture(0)
total_points = 0

RankLabel=("2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "A",
            "J",
            "K",
            "Q",
            )

SuitLabel=("club",
      "diamond",
      "heart",
      "spade",
)


class card_data:
    def __init__(self):
        self.contour = []
        self.width, self.height =  0,0
        self.corner_pts = []
        self.center = []
        self.warp = []
        self.rank = []
        self.suit = []
        self.rank_predict = "Unkown"
        self.suit_predict = "Unkown"
        self.rank_diff = 0
        self.suit_diff = 0


class Ranks:
    def __init__(self):
        self.img= []
        self.name = "rank"
        
class Suits:
    def __init__(self):
        self.img= []
        self.name = "suit"

def load_rank(filepath):
    train_ranks = []
    i = 0
    # for Rank in ['King','Two','Three','Four','Five','Six','Seven',
    #             'Eight','Nine','Ten','Jack','Queen','Ace']:
        
    for Rank in ['2','3','4','5','6','7',
                '8','9','10','J','Q','A','K',]:
        train_ranks.append(Ranks())
        train_ranks[i].name = Rank
        print(train_ranks[i].name)
        filename= Rank + '.png'
        train_ranks[i].img =cv2.imread(filepath+filename,cv2.IMREAD_GRAYSCALE)
        i=i+1
        
    return train_ranks

def load_suit(filepath):
    train_suits = []
    i = 0
    for Suit in ['club','diamond','spade','heart']:
        train_suits.append(Suits())
        train_suits[i].name = Suit
        filename= Suit + '.png'
        train_suits[i].img =cv2.imread(filepath+filename,cv2.IMREAD_GRAYSCALE)
        i=i+1
        
    return train_suits
        

def flatten(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    s = np.sum(pts, axis = 2)
    #nyari ujung paling kiri dan paling kanan corner dari kartu
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts,axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    #listing points corner, yang atas apabila vertical, yang bawah horizontal
    if w <= 0.8*h: 
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl
    if w >= 1.2*h: 
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8*h and w < 1.2*h:
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left
        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
    
    maxWidth = 200
    maxHeight = 300
    
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

    return warp


def process_card(contour, image):
    card = card_data()
    card.contour = contour

    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    card.corner_pts = pts
    
    x,y,w,h = cv2.boundingRect(contour)
    card.width, card.height = w,h
    
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    card.center = [cent_x, cent_y]
    
    card.warp = flatten(image, pts, w, h)
    cv2.imshow('kartu',card.warp)
    
    corner = card.warp[0:84, 0:32]
    corner_zoom = cv2.resize(corner,(0,0), fx=4, fy=4)
    cv2.imshow('corner',corner_zoom)
    
    white_level = corner_zoom[15,int((32*4)/2)]
    thresh_level = white_level - 30
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(corner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    
    rank = query_thresh[20:185, 0:128]
    suit = query_thresh[186:336, 0:128]
    
    rank_cnts, hier = cv2.findContours(rank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rank_cnts = sorted(rank_cnts, key=cv2.contourArea,reverse=True)
    
    if len(rank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(rank_cnts[0])
        rank_roi = rank[y1:y1+h1, x1:x1+w1]
        rank_sized = cv2.resize(rank_roi, (70,125), 0, 0) #ukurannya width sama height
        card.rank = rank_sized
        cv2.imshow('rank',card.rank)
        
    suit_cnts, hier = cv2.findContours(suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    suit_cnts = sorted(suit_cnts, key=cv2.contourArea,reverse=True)
    
    if len(suit_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(suit_cnts[0])
        suit_roi = suit[y2:y2+h2, x2:x2+w2]
        suit_sized = cv2.resize(suit_roi, (70, 100), 0, 0) #ukurannya width sama height
        card.suit = suit_sized
        cv2.imshow('suit',card.suit)

    return card

def draw(image,card):
    x = card.center[0]
    y = card.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)
    
    rank = card.rank_predict
    suit = card.suit_predict
    
    cv2.putText(image,(rank+' dari'),(x-60,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank+' dara'),(x-60,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    
    cv2.putText(image,suit,(x-60,y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit,(x-60,y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,200,200),2,cv2.LINE_AA)


def match_card(card):
    best_rank_diff = 0
    best_suit_diff = 0
    bestname_rank = "Unknown"
    bestname_suit = "Unknown"
    
    if (len(card.rank) != 0) and (len(card.suit) != 0):
        
        card_rank = cv2.cvtColor(card.rank,cv2.COLOR_GRAY2BGR)
        card_rank = cv2.resize(card_rank,(70,125))
        card_rank_normalized = card_rank / 255.0
        card_rank_input = np.expand_dims(card_rank_normalized, axis=0)
        card_rank_input = card_rank_input.astype('float32')
        
        
        possible_rank = modelRank.predict(card_rank_input,verbose=0)
        nR = np.max(np.where(possible_rank== possible_rank.max()))
        bestname_rank = RankLabel[nR]
        print(bestname_rank)
        best_rank_diff = nR

        
        card_suit = cv2.cvtColor(card.suit,cv2.COLOR_GRAY2BGR)
        card_suit = cv2.resize(card_suit,(70,100))
        card_suit_normalized = card_suit / 255.0
        card_suit_input = np.expand_dims(card_suit_normalized,axis=0)
        card_suit_input = card_suit_input.astype('float32')

        possible_suit = modelSuit.predict(card_suit_input,verbose=0)
        nS = np.max(np.where(possible_suit== possible_suit.max()))
        bestname_suit = SuitLabel[nS]       
        print(bestname_suit)
        best_suit_diff=nS

    # Return the identiy of the card and the quality of the suit and rank match
    return bestname_rank, bestname_suit, best_rank_diff, best_suit_diff
               
               
def calculate_blackjack_points(rank_predictions):
    rank_values = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
        'J': 10, 'Q': 10, 'K': 10, 'A': 11
    }
    
    rank_points = sum(rank_values.get(rank, 0) for rank in rank_predictions)
    
    
    num_aces = rank_predictions.count('A')
    
    # while total_points > 21 and num_aces:
    #     total_points -= 10  
    #     num_aces -= 1

    return rank_points

def closing(image, kernel_dilate, kernel_erosion):
    img_dilation = cv2.dilate(image, kernel_dilate, iterations=1) 
    img_erosion = cv2.erode(img_dilation, kernel_erosion, iterations=2)
    return img_erosion

kernel_dilasi = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]],dtype = np.uint8)

kernel_erosi = np.ones((5, 5), np.uint8)

#warna biru
lower_blue = np.array([90, 50, 0])
upper_blue = np.array([130, 255, 255])


#load rank sama suit
train_ranks = load_rank('matching/')
train_suits = load_suit('matching/')


#load model
modelRank = mCNN.LoadModel("bobotRank.h5")
modelSuit = mCNN.LoadModel("bobotSuit.h5")

while True:
    ret, frame = cap.read()
    
    process = cp.deepcopy(frame)
    
    hsv = cv2.cvtColor(process,cv2.COLOR_BGR2HSV)
    
    
    maskblue = cv2.inRange(hsv,lower_blue,upper_blue)
    foreground_mask = cv2.bitwise_not(maskblue)
    process = cv2.bitwise_and(process,process,mask=foreground_mask)
    
    foreground = closing(process,kernel_dilasi, kernel_dilasi)
    
    #Ngeblur biar threshold bagus
    frameGrayscaled = cv2.cvtColor(process, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frameGrayscaled, (3, 3), 0)
    
    #threholding shape (contour kartu)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    #nyari kartu dengan contour
    cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
    if len(cnts) == 0:
        cnts_sort =  []
        cnt_is_card = []
    
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])
        
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < 120000) and (size > 25000)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1
    
    
    if len(cnts_sort) != 0:
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                cards.append(process_card(cnts_sort[i],frame))

                # Find the best rank and suit match for the card.
                cards[k].rank_predict,cards[k].suit_predict,cards[k].rank_diff,cards[k].suit_diff = match_card(cards[k])
                
                
                # image = draw(frame, cards[k])
                k = k + 1
                
        # if(k>0):
        #     for i in range (k):
        #         total_points = calculate_blackjack_points(cards[k].rank_predict)
        
        
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                # total_points += calculate_blackjack_points(cards[i].rank_predict)
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(frame,temp_cnts, -1, (255,0,0), 2)
            
    

    # card_counter = f"Jumlah kartu ada {number_of_contours}"
    # cv2.putText(frame, card_counter, (50,50) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow('Result', frame)
    
    print("Total Points:", total_points)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()