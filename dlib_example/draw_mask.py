import cv2
import numpy as np

if __name__ == "__main__":
    # Create a black image, a window and bind the function to window
    file_name = "046.jpg"
    #img = cv2.imread('welcome_kacchan_top.jpg')
    img = cv2.imread(file_name)
    
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    
    #shape1 = np.loadtxt('katchan_men.txt')
    shape1 = np.load(file_name.split("/")[-1].split(".")[0]+"_mask.npy")
    
    for i in range(0,shape1.shape[0],2):
        cv2.rectangle(img, tuple(shape1[i][::-1]), tuple(shape1[i+1][::-1]), (0, 0, 255), -1)
        
    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    cv2.imwrite('wedding_out.jpg', img)
    #cv2.imwrite('katchan_out.jpg', img)
