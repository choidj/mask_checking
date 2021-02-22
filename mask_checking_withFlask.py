{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "joined-guest",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [12/Feb/2021 15:14:38] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:39] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:39] \"\u001b[37mGET /video_feed HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:40] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:51] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:53] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:55] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:57] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:14:59] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:01] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:03] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:05] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:07] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:09] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:11] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:13] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:15] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Feb/2021 15:15:17] \"\u001b[37mGET /get_result HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
    "from tensorflow.keras.models import load_model\n",
    "from flask import Flask, render_template, Response\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "import numpy as np\n",
    "\n",
    "app = Flask(__name__)       # 플라스크를 생성하고 app 변수에 flask 초기화 하여 실행\n",
    "\n",
    "facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')\n",
    "model = load_model('models/mask_detector.model')\n",
    "\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "maskResult = \"test\"\n",
    "\n",
    "def gen_frames():  \n",
    "    while True:\n",
    "        try:\n",
    "            success, frame = cap.read()  # read the camera frame\n",
    "            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
    "            h, w = frame.shape[:2]\n",
    "\n",
    "            blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))\n",
    "            facenet.setInput(blob)\n",
    "            dets = facenet.forward()\n",
    "    \n",
    "            for i in range(dets.shape[2]):\n",
    "                confidence = dets[0, 0, i, 2]\n",
    "                if confidence < 0.5:\n",
    "                    continue\n",
    "\n",
    "                x1 = int(dets[0, 0, i, 3] * w)\n",
    "                y1 = int(dets[0, 0, i, 4] * h)\n",
    "                x2 = int(dets[0, 0, i, 5] * w)\n",
    "                y2 = int(dets[0, 0, i, 6] * h)\n",
    "\n",
    "                face = frame[y1:y2, x1:x2]\n",
    "\n",
    "                face_input = cv2.resize(face, dsize=(224, 224))\n",
    "                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)\n",
    "                face_input = preprocess_input(face_input)\n",
    "                face_input = np.expand_dims(face_input, axis=0)\n",
    "\n",
    "                mask, nomask = model.predict(face_input).squeeze()\n",
    "\n",
    "                if mask > nomask:\n",
    "                    color = (0, 255, 0)\n",
    "                    label = 'Mask %d%%' % (mask * 100)\n",
    "                else:\n",
    "                    color = (0, 0, 255)\n",
    "                    label = 'No Mask %d%%' % (nomask * 100)\n",
    "\n",
    "                cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)\n",
    "                cv2.putText(frame, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)\n",
    "\n",
    "            if not success:\n",
    "                break\n",
    "            else:\n",
    "                ret, buffer = cv2.imencode('.jpg', frame)\n",
    "                frame = buffer.tobytes()\n",
    "                yield (b'--frame\\r\\n'\n",
    "                       b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')  # concat frame one by one and show result\n",
    "        except:\n",
    "            yield (\"get_frames실패\")\n",
    "#             print(\"get_frames실패\")\n",
    "#             continue\n",
    "            \n",
    "                \n",
    "            \n",
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/video_feed')\n",
    "def video_feed():\n",
    "    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')\n",
    "\n",
    "# @app.route('/get_result', methods=['GET'])\n",
    "# def get_result():\n",
    "#     return result\n",
    "\n",
    "@app.route('/test')\n",
    "def test():\n",
    "    return render_template('test.html')\n",
    "\n",
    "@app.route('/get_result', methods=['GET'])\n",
    "def get_result():\n",
    "    while True:\n",
    "        try:\n",
    "            success, frame = cap.read()  # read the camera frame\n",
    "            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
    "            h, w = frame.shape[:2]\n",
    "\n",
    "            blob = cv2.dnn.blobFromImage(frame, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))\n",
    "            facenet.setInput(blob)\n",
    "            dets = facenet.forward()\n",
    "            \n",
    "            for i in range(dets.shape[2]):\n",
    "                confidence = dets[0, 0, i, 2]\n",
    "                if confidence < 0.5:\n",
    "                    continue\n",
    "\n",
    "                x1 = int(dets[0, 0, i, 3] * w)\n",
    "                y1 = int(dets[0, 0, i, 4] * h)\n",
    "                x2 = int(dets[0, 0, i, 5] * w)\n",
    "                y2 = int(dets[0, 0, i, 6] * h)\n",
    "\n",
    "                face = frame[y1:y2, x1:x2]\n",
    "\n",
    "                face_input = cv2.resize(face, dsize=(224, 224))\n",
    "                face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)\n",
    "                face_input = preprocess_input(face_input)\n",
    "                face_input = np.expand_dims(face_input, axis=0)\n",
    "\n",
    "                mask, nomask = model.predict(face_input).squeeze()\n",
    "\n",
    "                if mask > nomask:\n",
    "                    color = (0, 255, 0)\n",
    "                    label = 'Mask %d%%' % (mask * 100)\n",
    "                    maskResult = \"성공\"\n",
    "                else:\n",
    "                    color = (0, 0, 255)\n",
    "                    label = 'No Mask %d%%' % (nomask * 100)\n",
    "                    maskResult = \"실패\"\n",
    "\n",
    "            if not success:\n",
    "                break\n",
    "            else:\n",
    "                return maskResult  # concat frame one by one and show result\n",
    "        \n",
    "        except:\n",
    "            return \"얼굴인식 실패\"\n",
    "#             print(\"얼굴인식 실패\")\n",
    "#             continue\n",
    "                \n",
    "#             cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)\n",
    "#             cv2.putText(frame, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)\n",
    "        \n",
    "#         if not success:\n",
    "#             break\n",
    "#         else:\n",
    "#             ret, buffer = cv2.imencode('.jpg', frame)\n",
    "#             frame = buffer.tobytes()\n",
    "#             return maskResult  # concat frame one by one and show result\n",
    "\n",
    "if __name__ == \"__main__\": \n",
    "   app.run()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}