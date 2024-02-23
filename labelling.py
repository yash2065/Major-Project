import os
import csv

# Path to your preprocessed images directory
preprocessed_dir = 'D:\Major Project\dataset/test'

# Path to the output labeling CSV file
csv_file = 'D:\Major Project\dataset/test\labeling.csv'

# Initialize the data list
data = []

# Iterate through each preprocessed image file in the directory
for filename in os.listdir(preprocessed_dir):
    if filename.endswith('.jpeg'):
        # Extract the label from the filename
        label = filename.split('.')[0]

        # Append the image filename and label to the data list
        data.append([filename, label])

# Write the data to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])  # Write the header
    writer.writerows(data)  # Write the data rows

print('Labeling CSV file created successfully!')