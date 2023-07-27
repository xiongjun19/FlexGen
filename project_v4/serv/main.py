from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import threading
import numpy as np
import pandas as pd
import json, os
from engineio.payload import Payload
Payload.max_decode_packets = 2048
# http://127.0.0.1:9980/#/screen1
# http://127.0.0.1:9980/#/screen2
# Store previous data samples
previous_cxl_data = None
previous_progress_data =None
previous_cxl_throughput_data =None
flag_cxl=False
flag_progress=False
flag_throughput = False
MAX_THROUGHPUT = 7 # IMPORTANT TO SHOW CIRCULAR PROGRESS INITIAL, BUT WILL BE AUTO CHANGE
MODE_SYNTHETIC = False
CXL_LENGTH=0
my_mode = 'cxl'

def get_mode_info():
    global my_mode
    message_path = '../../FlexGen/flexgen/message.txt'
    if os.path.exists(message_path):
        with open(message_path, 'r') as file:
            my_mode = file.read().strip()
    mode_list = ['cxl','disk','memverge','mem','mem1']
    for mode in mode_list:
        if mode in my_mode:
            return mode



def get_paths():
    MODE = get_mode_info()
    MODE_UPPER='CXL'
    try:
        MODE_UPPER = MODE.upper()
    except:
        print('NO RUNNING WORKLOAD FOUND!!!')
    # Actual One 
    history_cxl_filepath=f'../../FlexGen/flexgen/history-21july-b24/online_{MODE}.csv-gpu-0.csv'
    online_cxl_filepath=f'../../FlexGen/flexgen/online/online_{MODE}.csv-gpu-0.csv'
    history_disk_filepath=f'../../FlexGen/flexgen/history-21july-b24/online_disk.csv-gpu-0.csv'
    
    cxl_decode_throghput_filepath=f'../../FlexGen/flexgen/OPT-66b-{MODE_UPPER}-OUTPUT.log'
    disk_decode_throghput_filepath=f'../../FlexGen/flexgen/OPT-66b-DISK-OUTPUT.log'
    
    return history_cxl_filepath,online_cxl_filepath,history_disk_filepath,cxl_decode_throghput_filepath,disk_decode_throghput_filepath



if MODE_SYNTHETIC=='synthetic':
    # Synthetic One
    history_cxl_filepath='offline/history_cxl.csv'
    online_cxl_filepath='online/online_cxl.csv'
    history_disk_filepath='offline/history_disk.csv'
    disk_decode_throghput_filepath='../../FlexGen/project_v4/decode_throughput/disk.log'
    cxl_decode_throghput_filepath='../../FlexGen/project_v4/decode_throughput/cxl.log'
    

def extract_all_decode_throughputs(log_file_path):
    throughput_list = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'decode throughput' in line:
                throughput = float(line.split('decode throughput: ')[1].split(' token/s')[0])
                throughput_list.append(throughput)
    return throughput_list


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
def read_throughput_real(filepath):
    throughput_list = extract_all_decode_throughputs(filepath)
    if len(throughput_list) == 0:
        return '0'
    return throughput_list

def read_throughput(filepath):
    with open(filepath, "r") as file:
        # Read the single line of text
        text = file.readline()
    return text

# Print the t

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')




@socketio.on('data_request')
def handle_data_request(data_type,value=None):
    global previous_cxl_data, previous_progress_data,previous_cxl_throughput_data,flag_cxl,flag_progress,flag_throughput,CXL_LENGTH
    if data_type == 'cxl_history':
        # print('===================CXL HISTORY======================')
        history_cxl_filepath,_,_,_,_ = get_paths()
        data = read_csv_data(history_cxl_filepath)
    elif data_type == 'cxl_online':
        # print('===================CXL ONLINE======================')
        _,online_cxl_filepath,_,_,_ = get_paths()
        data = read_csv_data(online_cxl_filepath)
        if flag_cxl:
            if data.to_json() == previous_cxl_data.to_json():
                flag_cxl = True
                return
        previous_cxl_data = data
        
        
    elif data_type == 'disk_history':
        _,_,history_disk_filepath,_,_ = get_paths()
        data = read_csv_data(history_disk_filepath)
    elif data_type == 'live_progress_bar_value':
        if flag_progress==False:
            history_cxl_filepath,_,_,_,_= get_paths()
            history_cxl_data = read_csv_data(history_cxl_filepath)
            CXL_LENGTH = len(history_cxl_data)
        _,online_cxl_filepath,_,_,_= get_paths()
        cxl_data = read_csv_data(online_cxl_filepath)
        if flag_progress:
            if len(cxl_data) == previous_progress_data:
                flag_progress= True
                return
        previous_progress_data = len(cxl_data)
        percent = np.round(100*len(cxl_data)/CXL_LENGTH,1)
        if percent>100:
            percent=100
        
        print('MAX LENGTH',CXL_LENGTH)
        data = {"live_progress_bar_value_%":percent}
        data = json.dumps(data)
        
    elif data_type == 'live_cxl_throughput_value':
        # print('=========================================')
        _,_,_,cxl_decode_throghput_filepath,_= get_paths()
        data = get_throughput_data(data_type,cxl_decode_throghput_filepath)
        
        if flag_throughput:
            if previous_cxl_throughput_data == data:
                flag_throughput=True
                return
            previous_cxl_throughput_data=data
            
    elif data_type == 'disk_throughput_value':
        _,_,_,_,disk_decode_throghput_filepath= get_paths()
        data = get_throughput_data(data_type,disk_decode_throghput_filepath)
        
    elif data_type == 'cxl_online_row':
        _,online_cxl_filepath,_,_,_= get_paths()
        data = get_data_at_row_index(online_cxl_filepath,value)
    elif data_type == 'disk_online_row':
        
        _,_,history_disk_filepath,_,_= get_paths()
        data = get_data_at_row_index(history_disk_filepath,value)
    else:
        data = 'Invalid data type'
    if isinstance(data, pd.DataFrame):
        data = data.to_json(orient='records')
    emit('data_response', (data_type, data))

def get_throughput_data(data_type,filepath):
    global MAX_THROUGHPUT, SHOW_THROUGHPUT
    data_list = read_throughput_real(filepath)
    if data_type=='live_cxl_throughput_value':
        MAX_THROUGHPUT = max(data_list)+1
    print(f'MAX_THROUPUT: ',MAX_THROUGHPUT)
    data = str(data_list[-1]) # to get the lastest decode throughput
    if MODE_SYNTHETIC:
        data_list = read_throughput(filepath)
        data = data_list
    
    if data.find('\n'): 
        data = data.replace('\n','')
    
    data ={data_type:float(data),data_type+'%':100*float(data)/MAX_THROUGHPUT }
    json_data = json.dumps(data)

    return json_data
    

def get_data_at_row_index(filename,value):
    cxl_data_df =  pd.read_csv(filename)
    cxl_data_df['STEP'] = cxl_data_df.index
    try:
        row_data = cxl_data_df.loc[value, :]
        row_data = pd.DataFrame(row_data).T.reset_index(drop=True)
        row_data.columns = cxl_data_df.columns
        row_data = row_data.to_json(orient='records')
    except Exception as e:
        row_data = 'ROW INDEX NOT FOUND'
    data = f'ROW INDEX: {value}\n {row_data}'
    return data


def read_csv_data(filename):
    df = pd.read_csv(filename)
    df['STEP'] = df.index
    return df


if __name__ == '__main__':
    socketio.run(app,host='127.0.0.1', port=9980, debug=True)
