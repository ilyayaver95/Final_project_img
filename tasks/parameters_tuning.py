import pandas as pd
import os
from utils_skimage import return_params


photo_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 11, 12, 13, 14, 15, 16, 17, 18]
true_label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 good   \   0 bad

smallest_angle_arr = []
sum_of_angles_arr = []
diff_area_arr = []

photos_path = r'C:\Users\IlyaY\Desktop\לימודים\תשפג\ק\עיבוד תמונה\Lapis\all'

df = pd.DataFrame(columns=['ID', 'Smalles_angle', 'Sum_of_angles', 'Area_diff', 'Label'])
df['ID'] = photo_id
df['Label'] = true_label

for id in photo_id:  # create data set by labeled pencils from folder
    path = os.path.join(photos_path, str(id)+'.jpg')
    try:
        smallest_angle, sum_of_angles, diff_area = return_params(path)  # calculate the 3 features

        smallest_angle_arr.append(smallest_angle)
        sum_of_angles_arr.append(sum_of_angles)
        diff_area_arr.append(diff_area)

    except:
        print('id ', id)
        smallest_angle_arr.append(0)
        sum_of_angles_arr.append(0)
        diff_area_arr.append(0)

df['Smalles_angle'] = smallest_angle_arr
df['Sum_of_angles'] = sum_of_angles_arr
df['Area_diff'] = diff_area_arr

filepath = os.path.join(photos_path, 'data.csv')  # save data as csv
df.to_csv(filepath)




