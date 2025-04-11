import os
import boto3
import logging
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor

# Config the logger.
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REGION = 'ap-northeast-2'  # 원하는 리전으로 설정
KVS_STREAM_NAME = 'Ju-connect-catholic-test-contact-059dd53c-13de-4a5f-99b8-2beb715afe16'  # 스트림 이름 설정

class KvsAudioToWav:
    def __init__(self, save_directory):
        """Initialize the KVS clients and processor."""
        self.save_directory = save_directory
        self.kvs_fragment_processor = KvsFragementProcessor()  # Fragment Processor 초기화
        self.session = boto3.Session(region_name=REGION)
        self.kvs_client = self.session.client("kinesisvideo")
        self.kvs_media_client = None

    def get_media_client(self):
        """Get the KVS media client."""
        log.info('Getting KVS GetMedia Endpoint...')
        get_media_endpoint = self._get_data_endpoint(KVS_STREAM_NAME, 'GET_MEDIA')
        self.kvs_media_client = self.session.client('kinesis-video-media', endpoint_url=get_media_endpoint)

    def _get_data_endpoint(self, stream_name, api_name):
        """Helper method to get the data endpoint."""
        response = self.kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName=api_name
        )
        return response['DataEndpoint']

    def fetch_audio_and_save(self):
        """Fetch audio from Kinesis Video Stream and save as a wav file."""
        log.info(f'Fetching audio from stream: {KVS_STREAM_NAME}')
        get_media_response = self.kvs_media_client.get_media(
            StreamName=KVS_STREAM_NAME,
            StartSelector={'StartSelectorType': 'NOW'}
        )
        
        log.info('Processing fragments...')
        # 예시로 fragment_bytes 값을 설정하여 사용
        example_fragment_bytes = b'\x00\x01\x02\x03'  # 실제 데이터가 아닌 예시 값
        self.process_fragment(example_fragment_bytes)

    def process_fragment(self, fragment_bytes):
        """Process the fragment bytes to extract audio and save as wav."""
        try:
            # Save the fragment's audio track to a WAV file.
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)

            wav_file_name = 'audio_from_kinesis.wav'
            wav_file_path = os.path.join(self.save_directory, wav_file_name)

            log.info(f'Saving audio fragment as wav file: {wav_file_path}')
            with open(wav_file_path, 'wb') as wav_file:
                wav_file.write(fragment_bytes)

        except Exception as err:
            log.error(f"Error processing fragment: {err}")

if __name__ == "__main__":
    save_directory = '/home/user09/beaver/data/shared_files/amazon_kinesis/aws_kinesis/kinesis_test'
    kvs_audio = KvsAudioToWav(save_directory)
    kvs_audio.get_media_client()  # Media client 생성
    kvs_audio.fetch_audio_and_save()  # 오디오 데이터 받아서 저장

    # 예시 fragment_bytes 값으로 process_fragment 호출
    example_fragment_bytes = b'\x00\x01\x02\x03'  # 실제 데이터가 아닌 예시 값
    kvs_audio.process_fragment(example_fragment_bytes)

# ==================================================================================================
import os
import boto3
import logging
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import KvsFragementProcessor

# Config the logger.
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REGION = 'ap-northeast-2'  # 원하는 리전으로 설정
KVS_STREAM_NAME = 'Ju-connect-catholic-test-contact-059dd53c-13de-4a5f-99b8-2beb715afe16'  # 스트림 이름 설정

class KvsAudioToWav:
    def __init__(self, save_directory):
        """Initialize the KVS clients and processor."""
        self.save_directory = save_directory
        self.kvs_fragment_processor = KvsFragementProcessor()  # Fragment Processor 초기화
        self.session = boto3.Session(region_name=REGION)
        self.kvs_client = self.session.client("kinesisvideo")
        self.kvs_media_client = None

    def get_media_client(self):
        """Get the KVS media client."""
        log.info('Getting KVS GetMedia Endpoint...')
        get_media_endpoint = self._get_data_endpoint(KVS_STREAM_NAME, 'GET_MEDIA')
        self.kvs_media_client = self.session.client('kinesis-video-media', endpoint_url=get_media_endpoint)

    def _get_data_endpoint(self, stream_name, api_name):
        """Helper method to get the data endpoint."""
        response = self.kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName=api_name
        )
        return response['DataEndpoint']

    def fetch_audio_and_save(self):
        """Fetch audio from Kinesis Video Stream and save as a wav file."""
        log.info(f'Fetching audio from stream: {KVS_STREAM_NAME}')
        get_media_response = self.kvs_media_client.get_media(
            StreamName=KVS_STREAM_NAME,
            StartSelector={'StartSelectorType': 'NOW'}
        )
        
        log.info('Processing fragments...')
        # 예시로 fragment_bytes 값을 설정하여 사용
        example_fragment_bytes = b'\x00\x01\x02\x03'  # 실제 데이터가 아닌 예시 값
        self.process_fragment(example_fragment_bytes)
        return get_media_response

    def process_fragment(self, fragment_bytes):
        """Process the fragment bytes to extract audio and save as wav."""
        try:
            # Save the fragment's audio track to a WAV file.
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)

            wav_file_name = 'audio_from_kinesis.wav'
            wav_file_path = os.path.join(self.save_directory, wav_file_name)

            log.info(f'Saving audio fragment as wav file: {wav_file_path}')
            with open(wav_file_path, 'wb') as wav_file:
                wav_file.write(fragment_bytes)

        except Exception as err:
            log.error(f"Error processing fragment: {err}")

if __name__ == "__main__":
    save_directory = '/home/user09/beaver/data/shared_files/amazon_kinesis/aws_kinesis/kinesis_test'
    kvs_audio = KvsAudioToWav(save_directory)
    kvs_audio.get_media_client()  # Media client 생성
    example_fragment_bytes = kvs_audio.fetch_audio_and_save()  # 오디오 데이터 받아서 저장

    # 예시 fragment_bytes 값으로 process_fragment 호출
    # example_fragment_bytes = b'\x00\x01\x02\x03'  # 실제 데이터가 아닌 예시 값
    kvs_audio.process_fragment(example_fragment_bytes)
