from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
# from sklearn.cluster import DBSCAN
import os
from sklearn.preprocessing import scale
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import time
#loading the model

from keras.models import load_model
model = load_model('./keras-facenet/model/facenet_keras.h5')
detector = MTCNN()
# print(model.summary())
def test():
    # data=extract_faces("./uploads/suresh_Raina_dhoni_10.jpg")
    date = load_data("./uploads")
    result=do_clusterV2(date)
    for i in result:
        print(i["images"])
        print("#####")
       
#extract face from the single photograph
def extract_face(filename, required_size=(160, 160)):
    global detector
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def extract_faces(filename, required_size=(160, 160)):
    global detector
    faces=[]
    image = Image.open(filename)
  
    image = image.convert('RGB')
    pixels = np.asarray(image)
    
    results = detector.detect_faces(pixels)

    if len(results)==0:
        return faces
    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        faces.append(np.asarray(image))
    
    return faces

#extract all faces from all files in the 


# def load_data(folder):
#     all_faces=[]
#     for filename in os.listdir(folder):
#         path = folder+"/" + filename
#         faces = extract_faces(path)
#         print(len(faces))
#         all_faces.append({"id":path,"face_list":faces})
#     return all_faces

#get embeddings
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(axis=(1,2),keepdims=True),face_pixels.std(axis=(0,1),keepdims=True)
    face_pixels = (face_pixels - mean) / std
    yhat = model.predict(face_pixels)
    return yhat

def load_data(folder):
    global model
    all_faces=[]
    # print(os.listdir(folder))
    for filename in tqdm(os.listdir(folder)):
        
        path = folder+"/" + filename
        # print(path)
#         print("reading....",path)
        faces = extract_faces(path)
        if len(faces)>0:
          y_hat=get_embedding(model,np.asarray(faces))
        else:
          y_hat= []
        all_faces.append({"id":path,"face_list":faces,"face_embeddings":y_hat,"count":len(faces)})
    return all_faces

#l2 distance

def get_l2_distance(encoding1, encoding2):
  return np.linalg.norm(encoding2-encoding1)/100

#cosine distance
def get_cosine_distance(a,b):
  return dot(a, b)/(norm(a)*norm(b))

def do_cluster(all_faces):
    group_list=[]
    for image in all_faces:
        if len(group_list)==0:
            for embedding in image["face_embeddings"]:
                i,j= np.where(image["face_embeddings"]==embedding)      
                group_list.append({
                "images":[image["id"]],
                "center":embedding,
                "identity":image["face_list"][i[0]]
                   })
        else:
            if len(image["face_embeddings"])==0:
                group_list.append({
                    "images":[image["id"]],
                    "center":[],
                    "identity":None
                })
            else: 
                for embedding in image["face_embeddings"]:
                    i,j= np.where(image["face_embeddings"]==embedding)     
                    it=0
                    length=len(group_list)
       
                    matched=False
                    while it<length:
                        distance=get_cosine_distance(embedding,group_list[it]["center"])
          
                        if distance>0.6:
                            group_list[it]["center"]=embedding
                            matched=True
                            group_list[it]["images"].append(image["id"])
                        it=it+1
                    if matched==False:  
                           group_list.append({
                             "images":[image["id"]],
                                  "center":embedding,
                               "identity":image["face_list"][i[0]]
                            })
    return group_list
def add(list_name,imageid,center,identity):
  list_name.append({
      "images":imageid,
      "center":center,
      "identity":identity
  })
  return list_name
def do_clusterV2(all_faces):
     # assign one element to store results
    result=[]
    #loop through all image
    for image in all_faces:
        #check if grouplist is empty
        if len(result)==0:
        #then that is the first memeber of array
        #if face embedding is empty then add empty to result
            if len(image["face_embeddings"])==0:
                result =add(result,[image["id"]],None,None)
            else:
                #if it contains something add all embedding in the image to list
                for embedding in image["face_embeddings"]:
                    #get the row col index of the particular embedding vector
                    embedding_row,embedding_col = np.where(image["face_embeddings"]==embedding)
                    #add embedding in result
                    result = add(result,[image["id"]],[embedding],image["face_list"][embedding_row[0]])
        #if the group list is non zero then do the following
        else:
            #loop through all embedding in the image and compute the distance between list element and the particular image
            for embedding in image["face_embeddings"]:
            #get the index of embedding
                embedding_row,embedding_col = np.where(image["face_embeddings"]==embedding)
                #specify the iterator
                iterator=0
                #compute the length of result list
                length = len(result)
                #initialize matched variable as False
                matched =False
                #loop through result list and compute distance 
                while iterator < length:
                    #compute distance
                    distance=[]
                    #if center is single element
                    if len(result[iterator]["center"])<2:
                        distance.append(get_cosine_distance(embedding,result[iterator]["center"][0]))
                    else:
                        #if center is mutiple then loop through all center then check atleast single matches
                        for center in result[iterator]["center"]:
                            distance.append(get_cosine_distance(embedding,center))
          #             check with threshold
                    check=False
                    for dist in distance:
                        if dist>0.7:
                            check=True
                            break


                    if check==True:
                        result[iterator]["center"].append(embedding)
                        matched = True
                        result[iterator]["images"].append(image["id"])
                    iterator=iterator+1
            #if no embedding is matches add as new element in result array
                if matched==False:
                    result=add(result,[image["id"]],[embedding],image["face_list"][embedding_row[0]])
    return result
def make_cluster_folder(folder_name):
    pass




#all_faces = load_data("./data/data/")
#group_list =do_clusterV2(all_faces)
# all_faces= {"data":all_faces}
# data = { "data":data}
# print(data)
# import codecs, json 
# file_path="./data.json"
# json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


# len(group_list)
# array=[]
# i=0
# for i in range(len(group_list)):
#   array=[]
#   array.append(group_list[i]["identity"])
# #   plt.imshow(group_list[i]["identity"])
#   for j in range(len(group_list[i]["images"])):
#     image=Image.open(group_list[i]["images"][j])
#     image=image.convert("RGB")
#     image=np.array(image)
#     array.append(image)
#   array = np.array(array)
#   print("########################")
#   show_all_faces(array)import matplotlib.pyplot as plt

def show_all_faces(all_faces):
  i=0
  _ ,fig = plt.subplots(1, len(all_faces), figsize=(12,12))
  if len(all_faces)==1:
      plt.imshow(all_faces[0])
      return ""
  else:
      fig = fig.flatten()
      for f in fig:
        f.imshow(all_faces[i])
        i=i+1

# array=[]
# i=0
# for i in range(len(group_list)):
#   array=[]
#   array.append(group_list[i]["identity"])
# #   plt.imshow(group_list[i]["identity"])
#   for j in range(len(group_list[i]["images"])):
#     image=Image.open(group_list[i]["images"][j])
#     image=image.convert("RGB")
#     image=np.array(image)
#     array.append(image)
#   array = np.array(array)
#   print("########################")
#   show_all_faces(array)
# for i in group_list:
#   print(i["images"])
#   print("#####")
# test()