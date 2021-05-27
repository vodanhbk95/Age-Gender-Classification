import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')
print(data.head())
print(data.columns)
print(data.shape)

gender = []
for g in data['gender'].values:
    if g == 'male':
        gender.append(1)
    else:
        gender.append(0)

plt.hist(gender, range(3))
plt.title('There are total ' + str(len(gender) - sum(gender)) + ' female images and ' + str(sum(gender)) + ' male images')
plt.savefig('gender_dst_test.jpg')

plt.hist(data['age'], range(80))
plt.title('Age distribution')
plt.savefig('age_dst_test.jpg')