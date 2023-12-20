import os
import requests


def register(duration):
    pid = os.getpid()
    url = 'http://10.10.1.210/api/v1/job/create'
    data = {
            'student_id': '2120200498',
            'password': 'zhaoXIAOqun0521',
            'description': 'train',
            'server_ip': '10.10.1.210',
            'duration': duration,
            'pid': pid,
            'server_user': 'zhaoxiaoqun',
            # 'command': 'python',
            'use_gpu': 1,
            }
    r = requests.post(url, data=data)
    print(r.text)


# if __name__ == '__main__':
    # register('几小时')
