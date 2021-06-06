import numpy as np
import pandas as pd 

# afad = ['train', 'test', 'valid']
# for i, mode in enumerate(afad):
#     print(i, mode)
#     data = pd.read_csv(f'./afad_{mode}.csv')
#     data.iloc[:, 1:].to_csv(f'afad_{i}.csv', index=False)

df0 = pd.read_csv('test.csv')
df1 = pd.read_csv('test_afad.csv')
# df2 = pd.read_csv('afad_2.csv')

frame = [df0, df1]
result = pd.concat(frame)
result.to_csv('test_combinedataset.csv', index=False)

# data_root = '/data/asianface/tarball/AFAD-Full' 
# afad = pd.read_csv('da_afad.csv')
# afad_gender = []
# gender = afad['gender']
# # import ipdb; ipdb.set_trace()
# for i in range(len(gender)):
#     if gender[i] == 'male':
#         afad_gender.append(1)
#     elif gender[i] == 'female':
#         afad_gender.append(0)
        
# label_age = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
# reg_age = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
# age = afad['age']
# age_person = []
# for i in range(len(age)):
#     for j in range(len(label_age)):
#         if age[i] == label_age[j]:
#             age_person.append(reg_age[j])

# final_afad = np.vstack((data_root + '/' + afad['path'], age_person, afad_gender)).T
# final_afad_df = pd.DataFrame(final_afad)

# final_afad_df.to_csv('final_afad.csv', index=False)

# import cv2
# data = pd.read_csv('final_afad.csv')
# img_path = data['path']
# age = data['age']
# gender = data['gender']

# f_path = []
# f_age = []
# f_gender = []

# for i in range(len(data)):
#     image = cv2.imread(img_path[i])
#     if len(image.shape) == 3:
#         f_path.append(img_path[i])
#         f_age.append(age[i])
#         f_gender.append(gender[i])
#     else:
#         print(img_path[i])
    
#     print(i)
# final_data = np.vstack((f_path, f_age, f_gender)).T
# final_data_df = pd.DataFrame(final_data)
# final_data_df.to_csv('data_afad.csv', index=False)

# data = pd.read_csv('data_afad.csv')
# # import ipdb; ipdb.set_trace()
# data = data.sample(frac=1)
# data.to_csv('data_afad_v1.csv', index=False)
    

