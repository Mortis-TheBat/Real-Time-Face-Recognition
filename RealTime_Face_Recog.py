# Forked from- https://pythonprogramming.net/facial-recognition-python/
import face_recognition
import os, sys
from cv2 import cv2
from PIL import Image



KNOWN_FACES_DIR =   'C:/Users/Me/Desktop/Known_Faces/'
# UNKNOWN_FACES_DIR = 'C:/Users/Me/Desktop/Unknown_Faces'
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # 'cnn' - CUDA accelerated (if available) deep-learning pretrained model. HOG with lower end pcs

video = cv2.VideoCapture(0) #0,1,2,3

known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)

# Could prolly only process known  faces if there is a change in fir files
print("processing known faces")
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        print(filename)

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # (assuming one face per image as you can't be twice on one image)
        try: #For images where the system doesn't detect a face
            encoding = face_recognition.face_encodings(image)[0]
            # Append encodings and name
            known_faces.append(encoding)
            known_names.append(name)
        except:
            print("No face found in image: " + name+"/"+filename)  #Also add code to delete image
            print("Ateempting to remove the image")
            try:  #I'm getting some issues with os.remove. May work without try-catch
                os.remove(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            except:
                print("Failed to remove the image")



print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
while True:
    ret, image = video.read()
    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = [0,255,0]

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break