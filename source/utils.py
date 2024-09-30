from source.libraries import *
import os
import joblib
import shutil

class Save:
    def __init__(self, data, type, filename,  folder_name="output", timestamp=None):
        self.save(data, folder_name, filename, type, timestamp)

    def save(self, data, folder_name, filename, type, timestamp):
        if type == 'csv':
            if timestamp:
                filename = f"{filename}_{timestamp}.csv"
            else:
                filename = f"{filename}.csv"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            final_name = folder_name + '/' + filename
            data.to_csv(final_name, index=False)

        if type == 'png':
            if timestamp:
                filename = f"{filename}_{timestamp}.png"
            else:
                filename = f"{filename}.png"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            final_name = folder_name + '/' + filename
            data.savefig(final_name)  # Save the plot
        if type == 'model':
            if timestamp:
                filename = f"{filename}_{timestamp}.pkl"
            else:
                filename = f"{filename}.pkl"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            final_name = folder_name + '/' + filename
            joblib.dump(data, filename=final_name)

class Clear:
    def __init__(self):
        self.clear_folder('output')

    def clear_folder(self, folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove the file
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove the directory and all its contents
            except Exception as e:
                print(f"Error removing {item_path}: {e}")