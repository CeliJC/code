# Copyright 2019 Google, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START cloudrun_pubsub_server_setup]
# [START run_pubsub_server_setup]
import base64
import os
from compress import Compressor
from google.cloud import pubsub_v1
from google.cloud import bigquery
from flask import Flask, request
from lidar_camera_detector import *
from queue import *
import enviromentVariables as ev
from datetime import datetime
import threading
import ntplib


app = Flask(__name__)
lidar2cam = None

def inicizalization():
    print("-> Prediction process starts...")
    load_yolo()
    dir=os.getcwd()
    calib_file= dir + "/calibration_matlab_intrinsics.txt"
    lidar2cam = LiDAR2Camera(calib_file)
    print("Prediction inicialization starts...")
    ini_img= glob.glob(dir +"/ini_files/img/*.png")
    ini_pc=glob.glob(dir +"/ini_files/pc/*.pcd")
    for i in range(len(ini_img)):
        image = cv2.cvtColor(cv2.imread(ini_img[i]), cv2.COLOR_BGR2RGB)
        cloud = o3d.io.read_point_cloud(ini_pc[i])
        points= np.asarray(cloud.points)
        detector_image = lidar2cam.pipeline(image,points)
        print("step: ",i,"/",len(ini_img)-1)
    print("model inicialization completed...")
    return lidar2cam
lidar2cam = inicizalization()
#print(lidar2cam)
cliente_ntp = ntplib.NTPClient()

# [END run_pubsub_server_setup]
# [END cloudrun_pubsub_server_setup]



# [START cloudrun_pubsub_handler]
# [START run_pubsub_handler]
@app.route("/", methods=["POST"])
def index():
    global lidar2cam,cliente_ntp
    c = Compressor()
    c.use_bz2()
    envelope = request.get_json()
    if not envelope:
        msg = "no Pub/Sub message received"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    if not isinstance(envelope, dict) or "message" not in envelope:
        msg = "invalid Pub/Sub message format"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    pubsub_message = envelope["message"]

    if isinstance(pubsub_message, dict) and "data" in pubsub_message:
        print(".....message has arrived.....")
        #ts2 = cliente_ntp.request('europe.pool.ntp.org').tx_time
        dt2= datetime.now()
        ts2 = datetime.timestamp(dt2)
        ms_string={"data":pubsub_message["data"]}
        resized_image,point_cloud,ts3,ts1= decode(ms_string)
        ts4 = prediction(resized_image,point_cloud,lidar2cam)
        insert_data_bigquery(ts1,ts2,ts3,ts4)
        #print(ts2-ts1)
   
    return ("", 204)


# [END run_pubsub_handler]
# [END cloudrun_pubsub_handler]

def insert_data_bigquery(ts1,ts2,ts3,ts4):

    row_to_insert = [{"transference_time" : str(ts2-ts1),
                    "decode_time" : str(ts3-ts2),
                    "prediction_time" : str(ts4-ts3),
                    "total_time" : str(ts4-ts1)
    }]
        
    client = bigquery.Client()
    table_id = f"{ev.project_id}.{ev.bq_dataset}.{ev.bq_table}"
    errors = client.insert_rows_json(table_id, row_to_insert)  # Make an API request.
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))

def decode(ms):
    print("decode process starts...")
    c = Compressor()
    c.use_bz2()
    ms_decode = base64.b64decode(ms["data"])
    m=c.decompress(ms_decode)
    frame=eval(m.decode("utf-8"))
    point_cloud=np.array(frame["point_cloud"])
    camera_image=np.asarray(frame["camera_image"],dtype="uint8")
    image_decoded = cv2.imdecode(camera_image,1)
    resized_image = cv2.resize(image_decoded, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    #dt1 = datetime.strptime(frame["date_start"],'%Y-%m-%d %H:%M:%S.%f')
    ts1 = frame["timestamp1"]
    #ts3 = cliente_ntp.request('europe.pool.ntp.org').tx_time
    dt3= datetime.now()
    ts3 = datetime.timestamp(dt3)
    return resized_image,point_cloud,ts3,ts1



def prediction(resized_image,point_cloud,lidar2cam):
    #print(resized_image)
    #print(point_cloud)
    detector_image = lidar2cam.pipeline(resized_image,point_cloud)
    #ts4 = cliente_ntp.request('europe.pool.ntp.org').tx_time
    dt4= datetime.now()
    ts4 = datetime.timestamp(dt4)
    return ts4




if __name__ == "__main__":
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8080
    # This is used when running locally. Gunicorn is used to run the
    # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host="127.0.0.1", port=PORT, debug=True)

