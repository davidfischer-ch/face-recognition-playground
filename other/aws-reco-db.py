import boto3


def main():
    print(create_collection('souvenirs'))

def create_collection(reko, identifier):
    response = reko.create_collection(CollectionId=identifier)
    if response['StatusCode'] == 200:
        return response['CollectionArn']
    raise ValueError(response)

def index_faces(reko, collection, path, *, bucket=None):
    if bucket:
       image = {'S3Object': {'Bucket': bucket, 'Name': image}}
    else:
       with open(path, 'rb') as f:
           image = {'Bytes': f.read()}
    result = reko.index_faces(
        CollectionId=collection,
        DetectionAttributes=['ALL'],
        ExternalImageId=bucket + path,
        Image=image)
    raise ValueError(result)

if __name__ == '__main__':
    main()
