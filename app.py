import cv2
import csv
import os
import face_recognition
import datetime

known_faces = []
known_names = []

for filename in os.listdir('faces'):
    image = face_recognition.load_image_file(os.path.join('faces', filename))
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video source.")
else:
    attendance_marked = False

    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        while True:
            ret, frame = video_capture.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = 'Unknown'

                if True in matches:
                    matched_indices = [i for i, match in enumerate(matches) if match]
                    for index in matched_indices:
                        name = known_names[index]
                        recognized_names.append(name)

            if len(recognized_names) > 0 and not attendance_marked:
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                with open('attendance.csv', 'r') as read_file:
                    reader = csv.reader(read_file)
                    existing_names = set(row[0] for row in reader)

                for name in recognized_names:
                    if name not in existing_names:
                        writer.writerow([name, current_time])
                        existing_names.add(name)

                attendance_marked = True

            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
