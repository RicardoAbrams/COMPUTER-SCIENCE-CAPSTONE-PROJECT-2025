# SignSpeak: Sign to Text Translator

**SignSpeak** is a web-based application developed as a Capstone Project at Universidad Politécnica, designed to interpret sign language gestures and convert them into readable text in real time.

---

## 📌 Problem Description

Over 430 million people worldwide have disabling hearing loss, and this number is projected to increase to nearly 700 million by 2050. Fewer than 20% of hearing individuals understand sign language, creating a significant communication gap between the Deaf and Hard-of-Hearing community and non-signers. **SignSpeak** addresses this challenge by providing an intuitive, web-based solution that accurately translates between sign language and text to support inclusive communication.

---

## 🚀 Key Features

* **Real-Time Translation**: Captures video frames from the browser camera and displays translated text in real time.
* **Video Upload Support**: Allows users to upload video files for gesture analysis and interpretation.
* **Accessible User Interface**: Includes a clear navigation panel, authentication module (Login and Sign Up), and live camera controls.
* **Decoupled Architecture**: Features a modular separation between the Blazor frontend, client-side JavaScript camera logic, and FastAPI backend processing.

---

## 🛠️ Tech Stack

### Frontend
* **C# / Blazor (ASP.NET Core)**: Component-based framework used to build the interactive web UI.
* **HTML5 / CSS3 / JavaScript**: Used for responsive UI design and client-side video frame capture via browser APIs.

### Backend & AI
* **Python**: Primary language for backend logic and computer vision tasks.
* **FastAPI**: Asynchronous web framework used to build RESTful API endpoints.
* **MediaPipe**: Computer vision framework utilized for hand tracking and gesture detection.
* **Google Gemini API**: Integrated for optional AI analysis of uploaded video content.

### Data & Communication
* **Serialized Template Files**: Gesture recognition template data is stored locally in serialized files for efficient loading without requiring a traditional database for core functionality.
* **HTTP / JSON / FormData**: Secure communication protocols and formats used between the frontend and backend.
* **Git & GitHub**: Version control and team collaboration.

---

## 🏗️ System Architecture

The workflow of the SignSpeak system operates as follows:
1. **Frame Capture**: The Blazor web frontend and JavaScript camera logic capture live video frames from the user's browser.
2. **Transmission**: Frames are sent as REST HTTP requests (FormData / JSON) to the Python backend.
3. **Recognition**: FastAPI processes the frames using MediaPipe and local template files for gesture recognition.
4. **Optional AI Analysis**: Includes optional video analysis supported by the Google Gemini API.
5. **UI Update**: Recognized signs are returned and updated on the Blazor interface in real time.

---

## 👥 Team & Acknowledgments

### Authors
* **Victor M. Mejia Polanco** (#137219 COE-4022-80)
* **Ericka Ramirez** (#141885 COE-4022-80)
* **Dennis Y. Adorno Oyola** (#131708 CS4022-OL)
* **Ricardo Abrams Cortés** (#132510 CS-4022-80)
* **Sylvia G. Betancourt Díaz** (#141162 COE-4022-80)

### Advisor
* **Dr. Joanne Brenes Catinchi**

### Department & Institution
* **Universidad Politécnica** - Electrical and Computer Engineering and Computer Science Department

---

## 🔮 Future Work

* **Enhance Recognition Accuracy**: Refine computer vision algorithms for higher gesture precision.
* **Expand Vocabulary**: Increase the number of recognized signs and gestures.
* **Text-to-Speech (TTS) Integration**: Add speech synthesis to output spoken audio from translated text.
* **Mobile Optimization**: Optimize application responsiveness and performance for mobile devices.
* **Multilingual & Bidirectional Translation**: Expand language support and enable translation from text back to sign language.

---

## 📚 References

1. World Health Organization, *World Report on Hearing*, Geneva, Switzerland: WHO, 2021.
2. National Institute on Deafness and Other Communication Disorders, "American Sign Language," NIH, Bethesda, MD, USA, 2023.
3. S. K. Ong and Z. Wong, "Challenges in real-time sign language translation systems," *IEEE Access*, vol. 9, pp. 112345-112356, 2021.
