import boto3

if __name__ == "__main__":

    filename = 'input.jpg'
    reko = boto3.client('rekognition')

    with open(filename, 'rb') as f:
        response = reko.detect_labels(Image={'Bytes': f.read()})

    print('Detected labels in ' + imageFile)
    for label in response['Labels']:
         print (label['Name'] + ' : ' + str(label['Confidence']))
    print('Done...')
