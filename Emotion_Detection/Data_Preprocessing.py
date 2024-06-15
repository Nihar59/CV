import imghdr
import os
import cv2

# Define the list of acceptable image extensions
image_exts = ['jpeg', 'jpg', 'png']

# Path to the directory containing image classes and possibly other nested subdirectories
data_dir = '/content/train'

# Walk through all directories and files in the dataset
for root, dirs, files in os.walk(data_dir):
  for file in files:
    # Construct the path to the current file
    file_path = os.path.join(root, file)

    try:
      # Check the file type of the current file
      file_type = imghdr.what(file_path)

      # If the file extension is not in the allowed list, remove it
      if file_type not in image_exts:
        print(f'Image not in ext list {file_path}')
        os.remove(file_path)
      else:
        # Proceed to process the image if needed, for example, reading it with OpenCV
        img = cv2.imread(file_path)

    except Exception as e:
      # Print out the issue and the path of the problematic file
      print(f'Issue wth file {file_path}. Error: {e}')
      # Optionally, remove files that cause exceptions
      os.remove(file_path)
