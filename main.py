import cv2
from src.gesture.detector import HandRaiseDetector

detector = HandRaiseDetector()
cap = cv2.VideoCapture(0)

print("RaiseIt running — press ESC to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    raised_count = detector.detect(frame)

    cv2.putText(frame, f"Raised Hands: {raised_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("RaiseIt", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()