# COMPUTER-SCIENCE-CAPSTONE-PROJECT-2025

# SignSpeak: Sign to Text Translator[cite: 1]

**SignSpeak** is a web-based application developed as a Capstone Project at Universidad Politécnica, designed to interpret sign language gestures and convert them into readable text in real time[cite: 1].

---

## 📌 Problem Description[cite: 1]

Over 430 million people worldwide have disabling hearing loss, and this number is projected to increase to nearly 700 million by 2050[cite: 1]. Fewer than 20% of hearing individuals understand sign language, creating a significant communication gap between the Deaf and Hard-of-Hearing community and non-signers[cite: 1]. **SignSpeak** addresses this challenge by providing an intuitive, web-based solution that accurately translates between sign language and text to support inclusive communication[cite: 1].

---

## 🚀 Key Features[cite: 1]

* **Real-Time Translation**: Captures video frames from the browser camera and displays translated text in real time[cite: 1].
* **Video Upload Support**: Allows users to upload video files for gesture analysis and interpretation[cite: 1].
* **Accessible User Interface**: Includes a clear navigation panel, authentication module (Login and Sign Up), and live camera controls[cite: 1].
* **Decoupled Architecture**: Features a modular separation between the Blazor frontend, client-side JavaScript camera logic, and FastAPI backend processing[cite: 1].

---

## 🛠️ Tech Stack[cite: 1]

### Frontend
* **C# / Blazor (ASP.NET Core)**: Component-based framework used to build the interactive web UI[cite: 1].
* **HTML5 / CSS3 / JavaScript**: Used for responsive UI design and client-side video frame capture via browser APIs[cite: 1].

### Backend & AI
* **Python**: Primary language for backend logic and computer vision tasks[cite: 1].
* **FastAPI**: Asynchronous web framework used to build RESTful API endpoints[cite: 1].
* **MediaPipe**: Computer vision framework utilized for hand tracking and gesture detection[cite: 1].
* **Google Gemini API**: Integrated for optional AI analysis of uploaded video content[cite: 1].

### Data & Communication
* **Serialized Template Files**: Gesture recognition template data is stored locally in serialized files for efficient loading without requiring a traditional database for core functionality[cite: 1].
* **HTTP / JSON / FormData**: Secure communication protocols and formats used between the frontend and backend[cite: 1].
* **Git & GitHub**: Version control and team collaboration[cite: 1].

---

## 🏗️ System Architecture[cite: 1]

The workflow of the SignSpeak system operates as follows[cite: 1]:
1. **Frame Capture**: The Blazor web frontend and JavaScript camera logic capture live video frames from the user's browser[cite: 1].
2. **Transmission**: Frames are sent as REST HTTP requests (FormData / JSON) to the Python backend[cite: 1].
3. **Recognition**: FastAPI processes the frames using MediaPipe and local template files for gesture recognition[cite: 1].
4. **Optional AI Analysis**: Includes optional video analysis supported by the Google Gemini API[cite: 1].
5. **UI Update**: Recognized signs are returned and updated on the Blazor interface in real time[cite: 1].

---

## 👥 Team & Acknowledgments[cite: 1]

### Authors
* **Victor M. Mejia Polanco** (#137219 COE-4022-80)[cite: 1]
* **Ericka Ramirez** (#141885 COE-4022-80)[cite: 1]
* **Dennis Y. Adorno Oyola** (#131708 CS4022-OL)[cite: 1]
* **Ricardo Abrams Cortés** (#132510 CS-4022-80)[cite: 1]
* **Sylvia G. Betancourt Díaz** (#141162 COE-4022-80)[cite: 1]

### Advisor
* **Dr. Joanne Brenes Catinchi**[cite: 1]

### Department & Institution
* **Universidad Politécnica** - Electrical and Computer Engineering and Computer Science Department[cite: 1]

---

## 🔮 Future Work[cite: 1]

* **Enhance Recognition Accuracy**: Refine computer vision algorithms for higher gesture precision[cite: 1].
* **Expand Vocabulary**: Increase the number of recognized signs and gestures[cite: 1].
* **Text-to-Speech (TTS) Integration**: Add speech synthesis to output spoken audio from translated text[cite: 1].
* **Mobile Optimization**: Optimize application responsiveness and performance for mobile devices[cite: 1].
* **Multilingual & Bidirectional Translation**: Expand language support and enable translation from text back to sign language[cite: 1].

---

## 📚 References[cite: 1]

1. World Health Organization, *World Report on Hearing*, Geneva, Switzerland: WHO, 2021[cite: 1].
2. National Institute on Deafness and Other Communication Disorders, "American Sign Language," NIH, Bethesda, MD, USA, 2023[cite: 1].
3. S. K. Ong and Z. Wong, "Challenges in real-time sign language translation systems," *IEEE Access*, vol. 9, pp. 112345-112356, 2021[cite: 1].
