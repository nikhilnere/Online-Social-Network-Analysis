"""
sumarize.py
"""
import json
import re

def main():
    cluster_file = open('result_cluster.txt', 'r')
    result = json.loads(cluster_file.read())

    classify_file = open('result_classify.txt', 'r')
    result2 = json.loads(classify_file.read())

    result.update(result2)

    """for k, v in result.items():
        print (k, v)"""

    print ('Number of users collected : ' , result['Number of users collected : '])
    print ('Number of messages collected : ', result['Number of messages collected : '])
    print ('Number of communities discovred : ', result['Number of communities discovred : '])
    print ('Average number of users per community : ', result['Average number of users per community : '])
    print ('Number of positive instances : ', result['Number of positive instances : '])
    print ('Number of negative instances : ', result['Number of negative instances : '])
    print ('Example of positive instance : ', result['Example of positice instance : '])
    print ('Example of negative instance : ', result['Example of negative instance : '])

    summary_file = open("summary.txt", "w")
    summary_file.write('Number of users collected : ' + str(result['Number of users collected : ']))
    summary_file.write('\n')
    summary_file.write('Number of messages collected : ' + str(result['Number of messages collected : ']))
    summary_file.write('\n')
    summary_file.write('Number of communities discovred : ' + str(result['Number of communities discovred : ']))
    summary_file.write('\n')
    summary_file.write('Average number of users per community : ' + str(result['Average number of users per community : ']))
    summary_file.write('\n')
    summary_file.write('Number of positive instances : ' + str(result['Number of positive instances : ']))
    summary_file.write('\n')
    summary_file.write('Number of negative instances : ' + str(result['Number of negative instances : ']))
    summary_file.write('\n')
    summary_file.write('Example of positice instance : ' + result['Example of positice instance : '])
    summary_file.write('\n')
    summary_file.write('Example of negative instance : ' + result['Example of negative instance : '])
    summary_file.close()


if __name__ == '__main__':
    main()