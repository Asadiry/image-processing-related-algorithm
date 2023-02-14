import os
import cv2
from PIL import Image 
import numpy as np
from tqdm import tqdm

class EdgeDetectionOperator:
    """
        Usage: Edge Detection with Operator
        Args: 
            img: PIL.Image
        Return:
            edges: PIL.Image
    """

    def __init__(self):
        pass

    def __call__(self, img, opt_type="Sobel"):
        
        assert opt_type in ["Roberts", "Prewitt", "Sobel", "Scharr", "Laplacian", "Canny"]
        if opt_type == "Roberts":
            return self.Roberts_Operator(img)
        elif opt_type == "Prewitt":
            return self.Prewitt_Operator(img)
        elif opt_type == "Sobel":
            return self.Sobel_Operator(img)
        elif opt_type == "Scharr":
            return self.Scharr_Operator(img)
        elif opt_type == "Laplacian":
            return self.Laplacian_Operator(img)
        elif opt_type == "Canny":
            return self.Canny_Operator(img)

    def Roberts_Operator(self, img):
        img = np.array(img)

        kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=int)

        mask_x = cv2.filter2D(img, cv2.CV_16S, kernel_x)
        mask_y = cv2.filter2D(img, cv2.CV_16S, kernel_y)
        mask_x = cv2.convertScaleAbs(mask_x)
        mask_y = cv2.convertScaleAbs(mask_y)

        mask = cv2.addWeighted(mask_x, 0.5, mask_y, 0.5, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(np.uint8(mask))
    
    def Prewitt_Operator(self, img):
        img = np.array(img)

        kernel_x = np.array([[1,1,1], [0,0,0], [-1,-1,-1]], dtype=int)
        kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)

        mask_x = cv2.filter2D(img, cv2.CV_16S, kernel_x)
        mask_y = cv2.filter2D(img, cv2.CV_16S, kernel_y)
        mask_x = cv2.convertScaleAbs(mask_x)
        mask_y = cv2.convertScaleAbs(mask_y)

        mask = cv2.addWeighted(mask_x, 0.5, mask_y, 0.5, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(np.uint8(mask))

    def Sobel_Operator(self, img):
        img = np.array(img)

        mask_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        mask_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        mask_x = cv2.convertScaleAbs(mask_x)
        mask_y = cv2.convertScaleAbs(mask_y)

        mask = cv2.addWeighted(mask_x, 0.5, mask_y, 0.5, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(np.uint8(mask))

    def Scharr_Operator(self, img):
        img = np.array(img)

        mask_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        mask_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        mask_x = cv2.convertScaleAbs(mask_x)
        mask_y = cv2.convertScaleAbs(mask_y)

        mask = cv2.addWeighted(mask_x, 0.5, mask_y, 0.5, 0)
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(np.uint8(mask))

    def Laplacian_Operator(self, img, ksize=3):
        img = np.array(img)

        mask = cv2.Laplacian(img, cv2.CV_16S, ksize=ksize)
        mask = cv2.convertScaleAbs(mask)

        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(np.uint8(mask))

    def Canny_Operator(self, img, thresh1=50, thresh2=100):
        img = np.array(img)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        mask = cv2.Canny(img, thresh1, thresh2)

        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(np.uint8(mask))


def main():
    
    edge_opt = EdgeDetectionOperator()

    save_root = "./"
    load_root = "./"
    img_name = "demo"

    img_path = os.path.join(load_root, img_name+".jpg")
    img = Image.open(img_path).convert("L")
    
    mask = edge_opt.Sobel_Operator(img)
    save_path = os.path.join(save_root, img_name + "_sobel.png")
    mask.save(save_path)

    mask = edge_opt.Prewitt_Operator(img)
    save_path = os.path.join(save_root, img_name + "_prewitt.png")
    mask.save(save_path)

    mask = edge_opt.Roberts_Operator(img)
    save_path = os.path.join(save_root, img_name + "_roberts.png")
    mask.save(save_path)

    mask = edge_opt.Laplacian_Operator(img)
    save_path = os.path.join(save_root, img_name + "_laplacian.png")
    mask.save(save_path)

    mask = edge_opt(img, "Canny")
    save_path = os.path.join(save_root, img_name + "_canny.png")
    mask.save(save_path)

    mask = edge_opt.Scharr_Operator(img)
    save_path = os.path.join(save_root, img_name + "_scharr.png")
    mask.save(save_path)



if __name__ == '__main__':
    main()