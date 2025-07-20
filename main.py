
import cv2
import mediapipe as mp

mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1)
mpDraw=mp.solutions.drawing_utils

video=cv2.VideoCapture(0)

video.set(3, 1000)
video.set(4, 720)

def position_data(lmlist):
    global wrist, index_mcp, index_tip, midle_mcp, pinky_tip
    wrist = (lmlist[0][0], lmlist[0][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

def calculateDistance(p1, p2):
    x1,y1,x2,y2 = p1[0], p1[1], p2[0], p2[1]
    length = ((x2-x1)**2 + (y2-y1)**2) ** (1.0/2)
    return length

shiled=cv2.imread("america.png",-1)


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	bg_img = background_img.copy()
	

	img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)


	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	

	mask = cv2.medianBlur(a,5)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]

	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	

	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

	return bg_img

while True:
    ret,frame=video.read()
    img=cv2.flip(frame, 1)
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    # print(result.multi_hand_landmarks)
    lmList=[]
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            for id, lm in enumerate(handslms.landmark):
                h,w,c=img.shape
                coorx, coory=int(lm.x*w), int(lm.y*h)
                lmList.append([coorx, coory])
                # cv2.circle(img, (coorx, coory), 6, (0,255,0), -1)
                # mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS)
            position_data(lmList)
            palm=calculateDistance(wrist, index_mcp)
            distance=calculateDistance(index_tip, pinky_tip)
            ratio = distance/palm
            print(ratio)
            if ratio>1.2:
                centerX=midle_mcp[0]
                centerY=midle_mcp[1]
                shield_size=3.0
                diameter=round(palm*shield_size)

                x1 = round(centerX - (diameter/2))
                y1 = round(centerY - (diameter/2))

                h,w,c=img.shape

                if x1<0:
                    x1 = 0
                elif x1>w:
                    x1 = w

                if y1<0:
                    y1 = 0
                elif y1>h:
                    y1 = h
        
                if x1+diameter > w:
                    diameter = w-x1

                if y1+diameter > h:
                    diameter = h-y1

                shield_size = diameter,diameter
                if(diameter !=0):
                    img=overlay_transparent(img, shiled, x1, y1, shield_size)



    cv2.imshow("Frame",img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()