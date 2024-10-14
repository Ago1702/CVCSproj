import os
import re

def extract_number(filename):
    # Use regex to find the number in the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError("No number found in the filename.")

start_directory = '/work/cvcs2024/VisionWise/train'
files = sorted(os.listdir(start_directory))
print('Got all the file names')

new_index = 0
for file in files:
    if 'real' in file:
        break
    if 'fake' in file:
        old_index = extract_number(file)
        
        old_name_fake = os.path.join(start_directory,'train-fake-'+str(old_index).zfill(8) + '.png')
        new_name_fake = os.path.join(start_directory,'image-fake-'+str(new_index).zfill(8) + '.png')

        old_name_real = os.path.join(start_directory,'train-real-'+str(old_index).zfill(8) + '.png')
        new_name_real = os.path.join(start_directory,'image-real-'+str(new_index).zfill(8) + '.png')
        if not os.path.exists(old_name_real):
            os.remove(old_name_fake)
            continue

        #renaming part:
        os.rename(old_name_fake,new_name_fake)
        os.rename(old_name_real,new_name_real)

        new_index +=1
        if new_index % 10000 == 0:
            print(new_index)