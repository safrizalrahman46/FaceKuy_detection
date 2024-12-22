import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_1 = cv2.CascadeClassifier("face_1.xml")
camera = cv2.VideoCapture(1)

def face_detection(capture):
    optimized_capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)  
    faces = face_1.detectMultiScale(optimized_capture, scaleFactor=1.1, minNeighbors=5)
    return faces

def face_box(capture):
    faces = face_detection(capture)
    for x, y, w, h in faces:
        cv2.rectangle(capture, (x, y), (x + w, y + h), (0, 0, 255), 4)

def close_proses():
    camera.release()
    cv2.destroyAllWindows()
    exit()
    
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def main():
    while True:
        ret, capture = camera.read() 
        if not ret: 
            print("Failed to capture image")
            break
        
        face_box(capture)
        cv2.imshow("N Face", capture)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_proses()
        


if __name__ == '__main__':
    main()