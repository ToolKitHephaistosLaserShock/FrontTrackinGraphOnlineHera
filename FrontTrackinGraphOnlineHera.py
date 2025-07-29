import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Laurent Berthe @PIMM @CNRS
#Follow on video a front template located initialy by user in double 
# click left on image and circle size of Template Size
#Linear regression for constant velocity could be removed
#graph on line position, velocity front
#Inputs : 
#    FrameSpaceScale: space scale in Unit/px
#    DeltaTimeFrame : delta time between frame in time uit
#    VideoName: Name of Video file 
# On Camera window keyboard  
#'q' : quit and final graph
# p increase Template size  
# m decrease Template size  
# -> Frame advance
# <- Frame back
# Range : scale on image in um



#input
VideoName = "VideoFrontTest.avi"
DeltaTimeFrame = 100e-9 #s
FrameSpaceScale = 8.36 #um/pixel
Range=100 #(um)


def click(event, x, y, flags, param):
    global selected_point, template, current_frame_img
    if event == cv2.EVENT_LBUTTONDBLCLK:
        selected_point = (x, y)
        print(f"Selected Point : {selected_point}")
        frame = current_frame_img  # On utilise la variable globale mise à jour
        if frame is None:
            print("Erreur : frame empty")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, y0 = selected_point
        half = template_size // 2
        # Vérifier que la zone est dans l'image
        if y0 - half < 0 or y0 + half >= gray.shape[0] or x0 - half < 0 or x0 + half >= gray.shape[1]:
            print("Too closed the image border")
            return
        template = gray[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1].copy()
        print("Template to follow")

def GraphPreperation ():
    global fig,axs,line_pos_x,line_pos_y,regression_lineX,regression_lineY,line_vel_x,line_vel_y,line_vel_pos_x,line_vel_pos_y
    plt.ion()  # Mode interactif
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Prepare graph
    line_pos_x, = axs[0].plot([], [], 'r.-', label='X position')
    line_pos_y, = axs[0].plot([], [], 'b.-', label='Y position')
    regression_lineX, = axs[0].plot([], [], 'k--', label='Fit X(t)')
    regression_lineY, = axs[0].plot([], [], 'k--', label='Fit Y(t)')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position (um)")
    axs[0].set_title("Position vs Time")
    axs[0].legend()
    axs[0].grid(True)

    line_vel_x, = axs[1].plot([], [], 'r.-', label='Velocity X')
    line_vel_y, = axs[1].plot([], [], 'b.-', label='Velocity Y')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Velocity vs Time")
    axs[1].legend()
    axs[1].grid(True)

    line_vel_pos_x, = axs[2].plot([], [], 'r.-', label='X')
    line_vel_pos_y, = axs[2].plot([], [], 'b.-', label='Y')
    axs[2].set_xlabel("Position (um)")
    axs[2].set_ylabel("Velocities (m/s)")
    axs[2].set_title("Velocities vs Position")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    
#Initialisation for graph and data set
ParticuleXPositions = []
ParticuleYPositions = []
VelocityXPositions = []
VelocityYPositions = [] 
mean_Vx = 0
mean_Vy = 0
center_x, center_y = 100, 100
VelocityX, VelocityY =0,0
TimeFrame=[]

#Initialialisation of point selection.
selected_point = None
template = None
#Initial size of circle analysis front template
template_size = 30
positions = []
current_frame_img = None  # variable globale pour la frame actuelle

#Spacesale
longueur_pixels = int(Range/FrameSpaceScale)

x_start, y_start = 10, 240
x_end, y_end = x_start + longueur_pixels, y_start

#SI Unit
FrameSpaceScale=FrameSpaceScale*1e-6

#Graph preparation
GraphPreperation()

#Video reading
video = cv2.VideoCapture(VideoName)
TotalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video=cv2.VideoCapture(VideoName)
TotalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
WidthFrame = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
HeightFrame = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
FpsFrame = video.get(cv2.CAP_PROP_FPS)
DurationFrame = TotalFrames / FpsFrame if FpsFrame else 0
print ('Images properties')
print(f"Total frames : {TotalFrames}")
print(f"Dimensions : {WidthFrame}x{HeightFrame}")
print(f"FPS : {FpsFrame}")
print(f"Durée (s) : {DurationFrame:.2f}")


CurrentFrame = 0
cv2.namedWindow("Raw Video")

#Window Info
InfoWindow = np.ones((600,600, 3), dtype=np.uint8) * 255
cv2.imshow("Front-Tracking", InfoWindow)

#Point selection double clik on raw data
cv2.setMouseCallback("Raw Video", click)

#Loop on video raw - select action, front template, online velocity
while True:
    #move to CurrentFrame
    video.set(cv2.CAP_PROP_POS_FRAMES, CurrentFrame)
    #page blanche
    InfoWindow = np.ones((600, 700, 3), dtype=np.uint8) * 255
    ret, frame = video.read()
    if not ret:
        print("End.")
        break
    current_frame_img = frame.copy()  # mise à jour de la variable globale
    #Gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if template is not None:
        #search teamplate which is a front inside a circle define by double click on Raw image 
        res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
        #calculate new front/template position. 
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        top_left = min_loc
        center_x = top_left[0] + template_size // 2
        center_y = top_left[1] + template_size // 2
        positions.append((center_x, center_y))
        cv2.circle(frame, (center_x, center_y), template_size, (0, 0, 255), 2)

        if len(positions) >= 2:
            #calculate and save instantaneous velocity
            dx = (positions[-1][0] - positions[-2][0]) * FrameSpaceScale 
            dy = (positions[-1][1] - positions[-2][1]) * FrameSpaceScale 
            VelocityY = abs(dy/ DeltaTimeFrame)
            VelocityX = abs(dx/ DeltaTimeFrame)
            velocity = np.sqrt(dx**2 + dy**2) / DeltaTimeFrame
            ParticuleXPositions.append(center_x*FrameSpaceScale)
            ParticuleYPositions.append(center_y *FrameSpaceScale)
            VelocityXPositions.append(VelocityX)
            VelocityYPositions.append(VelocityY) 
            TimeFrame.append(CurrentFrame*DeltaTimeFrame)
            
            #Mean vleocities
            Vx = np.array(VelocityXPositions)
            Vy = np.array(VelocityYPositions)
            mean_Vx = np.mean(Vx)
            mean_Vy = np.mean(Vy)

            # Resultante
            mean_V = np.mean(np.sqrt(Vx**2 + Vy**2))
            print(f"Vitesse moyenne en X : {mean_Vx:.3e} m/s")
            print(f"Vitesse moyenne en Y : {mean_Vy:.3e} m/s")
            print(f"Vitesse moyenne résultante : {mean_V:.3e} m/s")
            line_pos_x.set_data(TimeFrame, ParticuleXPositions)
            line_pos_y.set_data(TimeFrame, ParticuleYPositions)
            #Linear regression on all dataX
            if len(ParticuleXPositions) >= 2:
                slopeX, interceptX = np.polyfit(TimeFrame, ParticuleXPositions, 1)
                TimeFrame_fit = np.linspace(min(TimeFrame), max(TimeFrame), 50)
                ParticuleXPositions_fit = slopeX * TimeFrame_fit + interceptX
                regression_lineX.set_data(TimeFrame_fit, ParticuleXPositions_fit)
                regression_lineX.set_label(f"Fit X: X = {slopeX:.2f}(m/s)Time + {interceptX:.2f}")
                axs[0].legend()
            if len(ParticuleYPositions) >= 2:
                slopeY, interceptY = np.polyfit(TimeFrame, ParticuleYPositions, 1)
                TimeFrame_fit = np.linspace(min(TimeFrame), max(TimeFrame), 50)
                ParticuleYPositions_fit = slopeY * TimeFrame_fit + interceptY
                regression_lineY.set_data(TimeFrame_fit, ParticuleYPositions_fit)
                regression_lineY.set_label(f"Fit Y: Y = {slopeY:.2f}(m/s)Time + {interceptY:.2f}")
                axs[0].legend()    
            
            #graph
            line_vel_x.set_data(TimeFrame, VelocityXPositions)
            line_vel_y.set_data(TimeFrame, VelocityYPositions)
            line_vel_pos_x.set_data(ParticuleXPositions, VelocityXPositions)
            line_vel_pos_y.set_data(ParticuleXPositions, VelocityYPositions)
            
            #Update template position
            cv2.circle(frame, (center_x, center_y), template_size, (0, 0, 255), 1)

            # AAxis auto-adjusting
            for ax in axs:
                ax.relim()
                ax.autoscale_view()
            # Refresh graph
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    #Print on infoWindow data set on-line
    cv2.putText(InfoWindow, f"Frame [Cmd :<-|->]: {CurrentFrame}/{TotalFrames}", (10,20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 1)
    cv2.putText(InfoWindow, f" Size [Cmd : p|m] : {template_size} px", (10, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
    
    cv2.putText(InfoWindow, f"ParticlePosition: (H : {center_x} px, V: {center_y} px)", (10, 70),
                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
     
    cv2.putText(InfoWindow, f" VelocityX : {VelocityX} ({mean_Vx:.3e})  m/s ", (10, 95),
                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
    cv2.putText(InfoWindow, f" VelocityY : {VelocityY} ({mean_Vy:.3e}) m/s " , (10, 120),
                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 2)
    
    cv2.putText(InfoWindow, f"FrameSpaceScale : {FrameSpaceScale} um/px", (10, 170),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 165, 255), 2)
    cv2.putText(InfoWindow, f"DeltaTimeFrame: {DeltaTimeFrame} s", (10, 195),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 165, 255), 2)
    cv2.putText(InfoWindow, "Video File : "+ VideoName, (10, 220),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 165, 255), 2)
    cv2.putText(InfoWindow, "Point selection : Dbl Left click on Raw" , (10, 245),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 2)
    cv2.putText(InfoWindow, "q to quit" , (10, 270),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 2)
    cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)  # vert, épaisseur 3
    cv2.putText(frame, f'{Range} um', (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.imshow("Front-Tracking", InfoWindow)
    cv2.imshow("Raw Video", frame)
    #Key open Command 
    key = cv2.waitKey(0)

    if key == ord('q'):
        print("End...")
        break
    elif key == 83:  # Arrow time down
        if CurrentFrame < TotalFrames - 1:
            CurrentFrame += 1
    elif key == 81:  # arrow time up
        if CurrentFrame > 0:
            CurrentFrame -= 1
    elif key == ord('p'): # increase template size
        template_size = min(250, template_size + 1)
        print(f" template_size +: {template_size}")
        continue
    elif key == ord('m'): #decrease template size
        template_size = max(1, template_size - 1)
        print(f" template_size -: {template_size}")
        continue

video.release()
cv2.destroyAllWindows()
