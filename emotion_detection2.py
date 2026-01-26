import cv2
from deepface import DeepFace

def main():
    cap = cv2.VideoCapture(0)

    print("Starting Emotion Detection... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            # DeepFace may return list or dict
            if isinstance(result, list):
                result = result[0]

            dominant_emotion = result["dominant_emotion"]

            # Display emotion on frame
            cv2.putText(
                frame,
                f"Emotion: {dominant_emotion}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("Error:", e)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
