'''
Convolutional Neural Network for Fashion Landmarks Detection.
Daniel E 201190945 for COMP592 Project due 20th of April
Download tfrecords files from https://1drv.ms/u/s!Au5dCZ4YubPGg4giMoXai4H2lLnBRA?e=fHdS9u
'''

#use this bulk_preprocess to preprocess images and update landmarks.json file.
#if dealing with csv: much quicker to convert landmark.csv to json with
#https://www.convertcsv.com/csv-to-json.htm
#use parseJSON to split json in small jsons corresponding to imgs in desired folder ,
#then create tfrecords file with this method
import json, csv, os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as Image
import tensorflow.compat.v1 as tf
#eager execution enabled by default in TF2.1.0


#change accordingly
MODE = "test"
basepath = "C:\\Users\\Nilan Delancourt\\Desktop\\projectenv\\dataset\\"
if (MODE == "train"):
    imagepath = basepath + "train\\"
    imagepathnew = basepath + "train2\\"
    datasetpath = basepath + "train\\"
    outputfile = basepath + "train.tfrecords"
    filenames = basepath + "train_files.csv"
    landmarkpath = basepath + "train_landmark.json"
    landmarkpathnew = basepath + "train_landmark2.json"

if (MODE == "val"):
    imagepath = basepath + "val\\"
    imagepathnew = basepath + "val2\\"
    datasetpath = basepath + "val\\"
    outputfile = basepath + "val.tfrecords"
    filenames = basepath + "val_files.csv"
    landmarkpath = basepath + "val_landmark.json"
    landmarkpathnew = basepath + "val_landmark2.json"

if (MODE == "test"):
    imagepath = basepath + "test\\"
    imagepathnew = basepath + "test2\\"
    datasetpath = basepath + "test\\"
    outputfile = basepath + "test.tfrecords"
    filenames = basepath + "test_files.csv"
    landmarkpath = basepath + "test_landmark.json"
    landmarkpathnew = basepath + "test_landmark2.json"

# Makes all images 256x256 by first centering them and filling in with white, 
# and resizing AND changes landmark files accordingly
def bulk_prepreprocess(imgnames_list, landmark_json_path):
    #fill  landmark_json_path = landmark_json_path
    def make_square(myimage, fill_color=(0, 0, 0)):
        x, y = myimage.size
        size = max(x, y, 512)
        new_image = Image.new("RGB", (size, size), fill_color)
        new_image.paste(myimage, (int((size - x) / 2), int((size - y) / 2)))
        return new_image

    #parse pandas dataset of image names to get imgnames, open image from dataset filepath
    #make it square, save then...
    #pd.
    with open(landmark_json_path, "r") as file:
        original_landmarks = json.load(file)
    new_landmarks = []
    i = 0
    for item in imgnames_list:
        sample_name = item[0]
        image_file_path = get_paths(sample_name, mode="imgpathonly")
        my_image = Image.open(image_file_path)
        #get size of image that"s just been loaded
        x, y = my_image.size
        #center image and fill rest with white
        new_image = make_square(my_image, fill_color=(255, 255, 255))
        #halve it
        new_size = (256,256)
        new_image = new_image.resize(new_size, resample=0)
        #path for a new folder, same name
        new_image_path = imagepathnew + sample_name + ".jpg"
        #save image to that path
        new_image.save(new_image_path)

        #update landmark in memory, filling in to 512,512 means add half of what was added
        #as half is added on either side, then halve landmark values because image dimensions 
        #were halved.
        #Update original landmarks
        new_dict = original_landmarks[i]
        for key, value in new_dict.items():
            if ("x" in key and value != 0):
                value = int((value + ((512 - x)/2))/2)
            elif ("y"in key and value != 0):
                value = int((value + ((512 - y)/2))/2)
            new_dict[key] = value
        #for each image i get a new dictionary of landmarks. Add it to the list.
        new_landmarks.append(new_dict)
        i+=1
        #break #do one only, so i can check
    
    #output list of new landmark dictionaries to json file for further processing
    with open(landmarkpathnew, "w+") as new_json:
        json.dump(new_landmarks, new_json, indent=1, separators=(",", ": "))

#get img names in a list then create json with name corresponding to landmark in big landmarks file
def splitJSON(imgnames_list, landmark_json_path):
    actual_results = []
    for arr in imgnames_list:
        for item in arr:
            actual_results.append(item)
    #above, done double list parsing because csv import is a list within a list 
    #counter to match name in array of imgnames with corresp landmark
    i = 0
    with open(landmark_json_path, "r") as file:
        data = json.load(file)
    for dict_ in data:
        img_name = actual_results[i]
        goto = imagepath + img_name + ".json"
        with open(goto, "w+") as newjsonfile:
            json.dump(dict_, newjsonfile, indent=1, separators=(",", ": "))
        i+=1

def _int64_feature_list(value):
    #Returns an int64_list from a bool / enum / int / uint.
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    #Returns a bytes_list from a string / byte.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_paths(sample_name, mode):
    image_file_path = os.path.join(imagepath + sample_name + ".jpg")
    landmark_file_path = os.path.join(datasetpath + sample_name + ".json")
    if mode=="both":
        #needed for writing to .tfrecords
        return image_file_path, landmark_file_path
    elif mode=="imgpathonly":
        #needed for bulk preprocessing
        return image_file_path
    else: raise Exception("A very specific bad thing happened.")

#Get feature dictionary
def get_pack(image_file_path, landmark_file_path):
    #split path and get filename.file, then split format away 
    filename = image_file_path.split("/")[-1].split(".")[-2]

    with tf.gfile.GFile(image_file_path, "rb") as imagefile:
        encoded_jpg = imagefile.read()
    
    with open(landmark_file_path) as lmfile:
        landmarks = json.load(lmfile)
        landmark_list = list(landmarks.values())
    return {"filename": filename,
            "image": encoded_jpg,
            "landmarks": landmark_list}

#Creates tf.example from json img pair using feature dict for further serialization to file
def create_tf_example(pack):
    feature = {
        "image/filename": _bytes_feature(pack["filename"].encode("utf8")),
        "image/encoded": _bytes_feature(pack["image"]),
        "features/labels": _int64_feature_list(pack["landmarks"])
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example

#repeat preprocessing process for each mode 
def main(_):
    #import image filenames, move to list - slower than csv but csv_read bugged
    with open(filenames, newline='') as csvfile:
        data = csv.reader(csvfile)
        imgnames = list(data)

    #UNCOMMENT THIS  to run bulk preprocess
    #bulk_prepreprocess(imgnames, landmarkpath)
    
    #UNCOMMENT this to run splitJSON
    #splitJSON(imgnames, landmarkpath)

    #UNCOMMENT this to run tfrecords merging/tf.example serialization
    with tf.io.TFRecordWriter(outputfile) as writer:
        j=0
        for row in imgnames:
            sample_name = row[0]
            img_file_path, landmark_file_path = get_paths(sample_name, mode="both")
            pack = get_pack(img_file_path, landmark_file_path)
            example = create_tf_example(pack)
            writer.write(example.SerializeToString())
            j+=1
            if (j % 10000==0):
                print("Processed 10000 samples.")

if __name__ == "__main__":
    tf.app.run()
