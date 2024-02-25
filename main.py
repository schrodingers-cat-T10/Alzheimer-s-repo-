import cv2
import pymongo
import face_recognition
from bson.binary import Binary
import numpy as np
import re
import streamlit as st
import speech_recognition as sr
from PIL import Image

# Function to extract information from speech
def extract_information(text):
    name_pattern = re.compile(r'(?i)\b(?:I\s+am\s+|my\s+name\s+is\s+)([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b')
    age_pattern = re.compile(r'\b(?:I\s+am\s+|my\s+age\s+is\s+)(\d{1,2})\b')
    profession_pattern = re.compile(r'\b(?:I\s+(?:work|study)\s+in\s+|I\s+am\s+a\s+|I\s+am\s+)(an?\s+)?([A-Za-z]+(?: [A-Za-z]+)*)\b')

    name_match = name_pattern.search(text)
    age_match = age_pattern.search(text)
    profession_match = profession_pattern.search(text)

    name = name_match.group(1) if name_match else None
    age = age_match.group(1) if age_match else None
    profession = profession_match.group(2) if profession_match else None

    return name, age, profession

# Function to recognize a person
def recognize_person(rgb_frame, collection):
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        cursor = collection.find({})
        for document in cursor:
            known_encoding = document.get('encoding')  # Get encoding or None if key doesn't exist
            if known_encoding:
                known_encoding = np.frombuffer(known_encoding, dtype=np.float64)
                matches = face_recognition.compare_faces([known_encoding], face_encoding)
                if matches[0]:
                    return document['_id'], document['name'], document['age'], document['profession'], document['image']
            
    return None, None, None, None, None

# Function to store a new person
def store_new_person(frame, rgb_frame, collection):
    r = sr.Recognizer()
    mic = sr.Microphone() 

    with mic as source:
        st.write("Please say your information...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        st.write("You said:", text)

        name, age, profession = extract_information(text)

        if name and age and profession:
            buffer = cv2.imencode('.jpg', frame)[1].tostring()
            image_binary = Binary(buffer)
            known_encoding = face_recognition.face_encodings(rgb_frame)[0]
            encoding_binary = Binary(known_encoding)
            collection.insert_one({"image": image_binary, "encoding": encoding_binary, "name": name, "age": age, "profession": profession})
            st.success("Information stored successfully.")
            return name, age, profession
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except sr.RequestError as e:
        st.error("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    return None, None, None

# MongoDB connection
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.1.5")
db = client["peopleIknow"]
collection = db["people"]

def main():
    st.title('Face Recognition System')

    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    person_id, name, age, profession, image_binary = recognize_person(rgb_frame, collection)

    if person_id:
        st.write("Person recognized! Person ID:", person_id)
        st.write("Name:", name)
        st.write("Age:", age)
        st.write("Profession:", profession)
        if image_binary:
            st.image(Image.open(io.BytesIO(image_binary)), caption='Recognized Person')
    else:
        st.write("Person not recognized.")

        # Store new person's information if not recognized
        name, age, profession = store_new_person(frame, rgb_frame, collection)
        if name:
            st.write("Name:", name)
            st.write("Age:", age)
            st.write("Profession:", profession)
            st.write("database updated successfully")

    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
