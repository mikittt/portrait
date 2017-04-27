import cv2
import numpy as np



# mouse callback function
def mouse_callback(event,x,y,flags,param):

    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        mask_point.append([y,x])

if __name__ == "__main__":
    
    global mask_point
    mask_point=[]
    file_name="046.jpg"
    # Create a black image, a window and bind the function to window
    img = cv2.imread(file_name)
    mask = np.zeros(img.shape)
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('image',mouse_callback, param=None)
    
    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            masks=[]
            for i in range(0,len(mask_point),2):
                parts=mask.copy()
                parts[mask_point[i][0]:mask_point[i+1][0],mask_point[i][1]:mask_point[i+1][1]]
                masks.append(parts)
            np.save(file_name.split("/")[-1].split(".")[0]+"_mask.npy",np.array(masks))
            break
    cv2.destroyAllWindows()
