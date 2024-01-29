from bs4 import BeautifulSoup
import numpy as np
from pydicom import read_file
from pathlib import Path
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm


def transform_dcm_to_npy(paths: list):
  """
  Function to transform DICOM files containing OCT-Scans of shape (z, x, y) into z .npy files of shape (x, y)
  
  Parameters
    ----------
    paths
        List containing .dcm paths to OCT scans
  """
  for path in paths:
    dc_path = Path(path) 
    pydicom_image = read_file(str(dc_path))
    oct_volume = np.array(pydicom_image.pixel_array)
    # we flip the ordering here because OCTExplorer's segmentation are exactly filpped (i.e. actually the dicom when opening is flipped)
    oct_volume = np.flip(oct_volume, axis=0)
    for i in range(oct_volume.shape[0]):
      target_path = path.split(".dcm")[0] + f"_{i}.npy"
      np.save(target_path, oct_volume[i])
  return 


def transform_segmentation_to_npy(paths_y: list):
    """
    Function to transform .xml files containing segmentations obtained from OCTExplorer of shape (z, x, y) into z .npy files of shape (x, y)
  
    Parameters
    ----------
    paths
        List containing .xml paths to segmentation of OCT scans
  """
    for path_y in paths_y:
        with open(path_y, 'r') as f:
            data = f.read()
        # Passing the stored data inside
        # the beautifulsoup parser, storing
        # the returned object
        Bs_data = BeautifulSoup(data, "html.parser")

        size = Bs_data.find_all("scan_characteristics")
        x_size = int(size[0].find("x").getText())
        y_size = int(size[0].find("y").getText())
        z_size = int(size[0].find("z").getText())

        segmentation = np.zeros([z_size, y_size, x_size])
        idx_y_previous=np.zeros([z_size, x_size])

        surface = Bs_data.find_all('surface')

        for idx_label, layer in enumerate(surface): # idx corresponds to class label
            z_enu = layer.find_all("bscan")
            for idx_z, b_scan in enumerate(z_enu):
                for idx_x, idx_y in enumerate(b_scan.find_all("y")):
                    idx_y_val = int(idx_y.getText())
                    segmentation[idx_z, int(idx_y_previous[idx_z, idx_x]):idx_y_val, idx_x] = idx_label
                    idx_y_previous[idx_z, idx_x] = idx_y_val

        for z_axis in range(segmentation.shape[0]):
            target_path = path_y.split(".xml")[0] + f"_{z_axis}.npy"
            np.save(target_path, segmentation[z_axis])    
    return 
        


# def transform_dcm_to_npy_3d(paths: list):
#     for path in paths:
#         dc_path = Path(path) #/ t.filename.iloc[1]
#         pydicom_image = read_file(str(dc_path))
#         oct_volume = np.array(pydicom_image.pixel_array)
#         # i in range(oct_volume.shape[0]):
#         target_path = path.split(".dcm")[0] + f"_3d_{i}.npy"
#         np.save(target_path, oct_volume[i])
#     return


# def transform_segmentation_to_npy_3d(paths_y: list):
#     for path_y in paths_y:
#         with open(path_y, 'r') as f:
#             data = f.read()
#         # Passing the stored data inside
#         # the beautifulsoup parser, storing
#         # the returned object
#         Bs_data = BeautifulSoup(data, "html.parser")

#         size = Bs_data.find_all("scan_characteristics")
#         x_size = int(size[0].find("x").getText())
#         y_size = int(size[0].find("y").getText())
#         z_size = int(size[0].find("z").getText())

#         segmentation = np.zeros([z_size, y_size, x_size])
#         idx_y_previous=np.zeros([z_size, x_size])

#         surface = Bs_data.find_all('surface')

#         for idx_label, layer in enumerate(surface): # idx corresponds to class label
#             z_enu = layer.find_all("bscan")
#             for idx_z, b_scan in enumerate(z_enu):
#                 for idx_x, idx_y in enumerate(b_scan.find_all("y")):
#                     idx_y_val = int(idx_y.getText())
#                     segmentation[idx_z, int(idx_y_previous[idx_z, idx_x]):idx_y_val, idx_x] = idx_label
#                     idx_y_previous[idx_z, idx_x] = idx_y_val

#         #for z_axis in range(segmentation.shape[0]):
#         target_path = "C:\\Users\\robin\\Desktop\\MASTER Mathematics in Data Science\\HiWi\\OCTdata\\gesunde_OCT\\3d_segmentation\\" + path_y.split("\\")[-1].split(".xml")[0] + f"_3d.npy" 
#         #np.save(target_path, segmentation)    
#     return segmentation   




def get_meta(meta_info, key):
    instance_name = [el.keyword for el in meta_info if key.lower() in str(el.keyword).lower()]
    if 0<len(instance_name)<=1:
        return getattr(meta_info, instance_name[0])
    elif len(instance_name)>1:
        print ("Multiple keys found: ", instance_name)
        values =  np.unique(np.array([getattr(meta_info, el) for el in instance_name]))
        if len(values) >1:
            print("Multiple instances for these keys: ", values)
            return None
        else:
            return values[0]
    else:
        print("No key found")
        return None
    
def try_open_dcm(pydicom_image, sitk_image, dc_path):

    try:
        oct_volume = np.array(pydicom_image.pixel_array)
        flag_pydicom_ok = True
    except:
        flag_pydicom_ok = False

    
    try:
        sitk_image.ReadImageInformation()
        image = sitk.ReadImage(str(dc_path))
        flag_sitk_ok = True
    except:
        flag_sitk_ok = False

    return flag_pydicom_ok, flag_sitk_ok


def get_all_dcm_meta(dc_paths):
    df = pd.DataFrame()
    for dc_path in tqdm(dc_paths):

        sitk_image_reader = sitk.ImageFileReader()
        # only read DICOM images
        sitk_image_reader.SetImageIO('GDCMImageIO')
        sitk_image_reader.SetFileName(str(dc_path))
        pydicom_reader = read_file(str(dc_path))
        row_dict ={el.keyword: getattr(pydicom_reader, el.keyword) for el in pydicom_reader if el.keyword!= "" and el.keyword!="PixelData"}
        patient_doctor_id = "".join(pydicom_reader.PatientName)
        row_dict["patient_id"] = patient_doctor_id.split("^")[0]
        row_dict["doctor_id"] = patient_doctor_id.split("^")[1]
        
        
        flag_pydicom_ok, flag_sitk_ok = try_open_dcm(pydicom_reader, sitk_image_reader, dc_path)
        row_dict["pydicom_readable"] = flag_pydicom_ok
        row_dict["sitk_readable"] = flag_sitk_ok
        row_dict["oct_readable"] = (flag_sitk_ok or flag_pydicom_ok)

        if flag_pydicom_ok or flag_sitk_ok:
            sitk_image_reader.ReadImageInformation()
            image = sitk.ReadImage(str(dc_path))
            depth = image.GetDepth()
            row_dict["num_slices"]= depth
            row_dict["img_size"]= image.GetSize()

        row_dict["filename"] = str(Path(dc_path).name)

        
        row = pd.DataFrame.from_records([row_dict])

        df = pd.concat([df, row], ignore_index=True)

    return df   

def rename_npy_files(x_y_map: dict, sep: str = "/", rename_file=False):
    """
    Helper function to rename files from dictionary which contains x_path as keys and y_path as values
    """
    # define names all by file name (which often looks like: surname_familyname_thirdname)
    names_all = list(set(["_".join(i.split(sep)[-1].split("_")[0:4]) for i in x_y_map.keys()]))
    
    # replace name by int
    replace_dict = {}
    for replace, name in enumerate(names_all):
        replace_dict[name] = replace

    x_y_map_updated = {}
    for file in x_y_map:
        val = x_y_map[file]
        val_new = val
        file_new = file
        for rep in replace_dict:
            val_new = val_new.replace(rep, str(replace_dict[rep]))
            file_new = file_new.replace(rep, str(replace_dict[rep]))
        x_y_map_updated[file_new] = val_new    
        if rename_file:
            os.rename(file, file_new)
            os.rename(val, val_new)
        
    return x_y_map_updated, replace_dict

## Can be run as follwed:
if __name__ == "__main__":
    directory = "C:\\Users\\robin\\Downloads\\gesunde_OCT_verbessert\\gesunde OCT verbessert"
    x_path = []
    y_path = []

    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".dcm") and "Final" not in filename and not filename.startswith(".") and "Spectralis" in filename: 
            #transform_dcm_to_tensor(directory + file)
            #  print(filename)
            x_path.append(directory + "\\" +filename)
            continue
        elif filename.endswith(".xml") and "Final" in filename and "Surfaces" in filename:
            y_path.append(directory + "\\" + filename)

    #transform_dcm_to_npy(x_path)
    #transform_segmentation_to_npy(y_path)
    