import csv
import os
## random utils
def safe_divide(current, prev):
    try:
        if prev:
            return current / prev
        else:
            return 0.0
    except ZeroDivisionError:
        print(current)
        print(prev)
        return 0.0

def save_dictionary(dictionary, dir, name):
    full_dir = os.path.join(os.getcwd(), dir)

    try:
        os.makedirs(full_dir)
    except:
        print('Dictionary folder exists overwriting')
    full_file_path = os.path.join(full_dir, name)
    with open(full_file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key,value in dictionary.items():
            writer.writerow([key,value])


