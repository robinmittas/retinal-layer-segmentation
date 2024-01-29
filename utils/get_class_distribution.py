import numpy as np

def get_class_distribution(x_y_map):
    """
    Function to calculate the average number of pixels per class in all images
    
    Parameters
    ----------
    x_y_map
        Dict containing paths to input images as keys and paths to segmentation images as values  
    """
    class_0 = []
    class_1 = []
    class_2 = []
    class_3 = []

    for i in x_y_map:
        y_data = np.load(x_y_map[i])
        # replace retinal layers (we just use 4 retinal layers instead of all 4)
        replace = {2:1, 3:1, 4:1, 5:1,
                   6:2,  
                   7:3, 8:3, 9:3, 10:3, 11:3}
        for key in replace:
            y_data = np.where(y_data==key, replace[key], y_data)
        y_data_pixels = np.prod(y_data.shape)
        class_0.append(np.array(y_data==0).sum() / y_data_pixels)
        class_1.append(np.array(y_data==1).sum() / y_data_pixels)
        class_2.append(np.array(y_data==2).sum() / y_data_pixels)
        class_3.append(np.array(y_data==3).sum() / y_data_pixels)
        
    return np.mean(class_0), np.mean(class_1), np.mean(class_2), np.mean(class_3)
    