import cv2
from src.gesture.detector import HandRaiseDetector

COLORS = {
    'confirmed': (0, 255, 0),    # green
    'raising':   (0, 255, 255),  # yellow
    'countdown': (0, 165, 255),  # orange
    'idle':      (255, 0, 0),    # blue
}

detector = HandRaiseDetector()
cap = cv2.VideoCapture(0)

print("RaiseIt running — press ESC to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    raised_count = detector.detect(frame)

    # Draw a circle and label for each tracked person
    for person in detector.tracked_people:
        center = (int(person['center'][0]), int(person['center'][1]))
        state = person['state']
        color = COLORS.get(state, (255, 0, 0))

        if state == 'confirmed':
            cv2.circle(frame, center, 20, color, -1)  # filled green

        elif state == 'raising':
            cv2.circle(frame, center, 15, color, 2)   # hollow yellow
            progress = f"{person['frames_raised']}/60"
            cv2.putText(frame, progress, (center[0] - 20, center[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        elif state == 'countdown':
            cv2.circle(frame, center, 20, color, 2)   # hollow orange
            countdown = f"DOWN: {30 - person['frames_not_raised']}"
            cv2.putText(frame, countdown, (center[0] - 30, center[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        else:
            cv2.circle(frame, center, 5, color, -1)   # small blue dot

    # Show raised hand count
    cv2.putText(frame, f"Raised Hands: {raised_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("RaiseIt", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()